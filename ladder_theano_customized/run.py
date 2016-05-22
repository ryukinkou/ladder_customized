#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import logging
import os
import subprocess
from argparse import ArgumentParser, Action
from collections import OrderedDict
import sys

import numpy
import time
import theano
from theano.tensor.type import TensorType

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import PARAMETER
from fuel.datasets import MNIST, CIFAR10
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer

from picklable_itertools import cycle, imap
from itertools import izip, product, tee

logger = logging.getLogger('main')

from utils import ShortPrinting, prepare_dir, load_df, DummyLoop
from utils import SaveExpParams, SaveLog, SaveParams, AttributeDict
from nn import ZCA, ContrastNorm
from nn import ApproxTestMonitoring, FinalTestMonitoring, TestMonitoring
from nn import LRDecay
from ladder import LadderAE


# 属性类dict，底层是dict
class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


# 创建目录
def prepare_dir(save_to, results_dir='results'):
    # 合并两个文件／目录路径
    base = os.path.join(results_dir, save_to)
    # 目录名 + 数字的方式为目录命名，创建若失败，数字+1继续创建直到成功
    suffix = 0
    while True:
        name = base + str(suffix)
        try:
            os.makedirs(name)
            break
        except OSError:
            suffix += 1
    return name


# 读取、记录参数
def load_and_log_params(cli_params):
    cli_params = AttributeDict(cli_params)
    # 如果有load_from参数，说明参数是有文件里面读取的，文件格式为hdf
    if cli_params.get('load_from'):
        # load_from值 + params组成完整地址
        # string => dict
        p = load_df(cli_params.load_from, 'params').to_dict()[0]
        # dict => AttributeDict
        p = AttributeDict(p)

        for key in cli_params.iterkeys():
            if key not in p:
                p[key] = None
        new_params = cli_params
        loaded = True

    # 如果没有load from参数，直接封装一下cli_params
    else:
        p = cli_params
        new_params = {}
        loaded = False

        # Make dseed seed unless specified explicitly
        # dseed为空而seed不为空时，dseed复制为seed
        if p.get('dseed') is None and p.get('seed') is not None:
            p['dseed'] = p['seed']

    # log相关
    logger.info('== COMMAND LINE ==')
    logger.info(' '.join(sys.argv))

    logger.info('== PARAMETERS ==')
    for k, v in p.iteritems():
        if new_params.get(k) is not None:
            p[k] = new_params[k]
            replace_str = "<- " + str(new_params.get(k))
        else:
            replace_str = ""
        logger.info(" {:20}: {:<20} {}".format(k, v, replace_str))
    return p, loaded


# 设置数据
def setup_data(p, test_set=False):

    # CIFAR10与MNIST都是封装过后的HDF5数据集
    # p.dataset为命令行传入的参数，在cifar10与mnist之间选择其一
    dataset_class, training_set_size = {
        'cifar10': (CIFAR10, 40000),
        'mnist': (MNIST, 50000),
    }[p.dataset]

    # 可以通过命令行指定为标注样本的大小
    # Allow overriding the default from command line
    if p.get('unlabeled_samples') is not None:
        training_set_size = p.unlabeled_samples

    # 选出mnist数据集里面的train子集
    train_set = dataset_class("train")

    # Make sure the MNIST data is in right format
    # 对minst进行数据检查，查看是否所有值都在0-1之间且都为float
    if p.dataset == 'mnist':
        # features大小为60000＊1＊28＊28，num_examples*channel*height*weight，minst为灰度图片所以channel=1
        d = train_set.data_sources[train_set.sources.index('features')]
        assert numpy.all(d <= 1.0) and numpy.all(d >= 0.0), \
            'Make sure data is in float format and in range 0 to 1'

    # 随机打乱样本顺序
    # Take all indices and permutate them
    all_ind = numpy.arange(train_set.num_examples)
    if p.get('dseed'):
        # 通过dseed制作一个随机器，用于打乱样本编号
        rng = numpy.random.RandomState(seed=p.dseed)
        rng.shuffle(all_ind)

    d = AttributeDict()

    # Choose the training set
    d.train = train_set
    # 此时index应该都被打乱
    # 取出前training_set_size个数的样本做为训练集（的index）
    d.train_ind = all_ind[:training_set_size]

    # 选出一部分数据作为验证集
    # Then choose validation set from the remaining indices
    d.valid = train_set
    # 全部的数据集中去掉训练用的样本，剩下的作为验证集
    d.valid_ind = numpy.setdiff1d(all_ind, d.train_ind)[:p.valid_set_size]

    logger.info('Using %d examples for validation' % len(d.valid_ind))

    # 如果有测试数据的话，生成测试数据的index
    # Only touch test data if requested
    if test_set:
        d.test = dataset_class("test")
        d.test_ind = numpy.arange(d.test.num_examples)

    # Setup optional whitening, only used for Cifar-10
    # 计算特征值的维度，shape[1:]：获取第一个样本的维度
    in_dim = train_set.data_sources[train_set.sources.index('features')].shape[1:]
    if len(in_dim) > 1 and p.whiten_zca > 0:
        assert numpy.product(in_dim) == p.whiten_zca, \
            'Need %d whitening dimensions, not %d' % (numpy.product(in_dim),
                                                      p.whiten_zca)

    # 归一化参数如果不为空，创建归一化类
    cnorm = ContrastNorm(p.contrast_norm) if p.contrast_norm != 0 else None

    def get_data(d, i):
        data = d.get_data(request=i)[d.sources.index('features')]

        # Fuel provides Cifar in uint8, convert to float32
        # 检查data集合中的item是否符合float32类型
        data = numpy.require(data, dtype=numpy.float32)
        # TODO ContrastNorm.apply
        return data if cnorm is None else cnorm.apply(data)

    if p.whiten_zca > 0:
        logger.info('Whitening using %d ZCA components' % p.whiten_zca)
        # TODO ZCA
        whiten = ZCA()
        whiten.fit(p.whiten_zca, get_data(d.train, d.train_ind))
    else:
        whiten = None

    return in_dim, d, whiten, cnorm


def setup_model(p):
    ladder = LadderAE(p)
    # Setup inputs
    input_type = TensorType('float32', [False] * (len(p.encoder_layers[0]) + 1))
    x_only = input_type('features_unlabeled')
    x = input_type('features_labeled')
    y = theano.tensor.lvector('targets_labeled')
    ladder.apply(x, y, x_only)

    # Load parameters if requested
    if p.get('load_from'):
        with open(p.load_from + '/trained_params.npz') as f:
            loaded = numpy.load(f)
            cg = ComputationGraph([ladder.costs.total])
            current_params = VariableFilter(roles=[PARAMETER])(cg.variables)
            logger.info('Loading parameters: %s' % ', '.join(loaded.keys()))
            for param in current_params:
                assert param.get_value().shape == loaded[param.name].shape
                param.set_value(loaded[param.name])

    return ladder


def unify_labels(y):
    """ Work-around for Fuel bug where MNIST and Cifar-10
    datasets have different dimensionalities for the targets:
    e.g. (50000, 1) vs (60000,) """
    yshape = y.shape
    # 将多维数据扁平化成一维数据
    y = y.flatten()
    assert y.shape[0] == yshape[0]
    return y

def make_datastream(dataset, indices, batch_size,
                    n_labeled=None, n_unlabeled=None,
                    balanced_classes=True, whiten=None, cnorm=None,
                    scheme=ShuffledScheme):
    if n_labeled is None or n_labeled == 0:
        n_labeled = len(indices)
    if batch_size is None:
        batch_size = len(indices)
    if n_unlabeled is None:
        n_unlabeled = len(indices)

    # 未标记数据要少于标记数据
    assert n_labeled <= n_unlabeled, 'need less labeled than unlabeled'

    if balanced_classes and n_labeled < n_unlabeled:
        # Ensure each label is equally represented
        logger.info('Balancing %d labels...' % n_labeled)
        all_data = dataset.data_sources[dataset.sources.index('targets')]
        y = unify_labels(all_data)[indices]
        # class的数量为y+1
        n_classes = y.max() + 1

        print(n_classes)

        assert n_labeled % n_classes == 0
        # 每个类有多少个样本都是一样的
        n_from_each_class = n_labeled / n_classes

        i_labeled = []
        for c in range(n_classes):
            i = (indices[y == c])[:n_from_each_class]
            i_labeled += list(i)
    else:
        i_labeled = indices[:n_labeled]

    # Get unlabeled indices
    i_unlabeled = indices[:n_unlabeled]

    # ds = SemiDataStream(
    #     data_stream_labeled=Whitening(
    #         DataStream(dataset),
    #         iteration_scheme=scheme(i_labeled, batch_size),
    #         whiten=whiten, cnorm=cnorm),
    #     data_stream_unlabeled=Whitening(
    #         DataStream(dataset),
    #         iteration_scheme=scheme(i_unlabeled, batch_size),
    #         whiten=whiten, cnorm=cnorm)
    # )

    ds = 0

    return ds



# 训练分类器
def train(cli_params):
    cli_params['save_dir'] = prepare_dir(cli_params['save_to'])

    # log设定相关，无视 START
    logfile = os.path.join(cli_params['save_dir'], 'log.txt')
    # Log also DEBUG to a file
    fh = logging.FileHandler(filename=logfile)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.info('Logging into %s' % logfile)
    # log设定相关，无视 END

    # 读取并记录命令行参数，参数都塞入p里面
    p, loaded = load_and_log_params(cli_params)

    # 根据参数和环境变量FUEL_DATA_PATH读取数据
    in_dim, data, whiten, cnorm = setup_data(p, test_set=False)

    #
    if not loaded:
        # Set the zero layer to match input dimensions
        p.encoder_layers = (in_dim,) + p.encoder_layers

    ladder = setup_model(p)

    # Training
    all_params = ComputationGraph([ladder.costs.total]).parameters
    logger.info('Found the following parameters: %s' % str(all_params))

    # all_params里面有一堆参数名类似的东西具体还不知道有什么作用

    # 用于batch norm的参数
    # Fetch all batch normalization updates. They are in the clean path.
    bn_updates = ComputationGraph([ladder.costs.class_clean]).updates
    assert 'counter' in [u.name for u in bn_updates.keys()], \
        'No batch norm params in graph - the graph has been cut?'

    training_algorithm = GradientDescent(
        cost=ladder.costs.total, params=all_params,
        step_rule=Adam(learning_rate=ladder.lr))
    # In addition to actual training, also do BN variable approximations
    # batchnorm参数近似
    training_algorithm.add_updates(bn_updates)

    short_prints = {
        "train": {
            'T_C_class': ladder.costs.class_corr,
            'T_C_de': ladder.costs.denois.values(),
        },
        "valid_approx": OrderedDict([
            ('V_C_class', ladder.costs.class_clean),
            ('V_E', ladder.error.clean),
            ('V_C_de', ladder.costs.denois.values()),
        ]),
        "valid_final": OrderedDict([
            ('VF_C_class', ladder.costs.class_clean),
            ('VF_E', ladder.error.clean),
            ('VF_C_de', ladder.costs.denois.values()),
        ]),
    }

    main_loop = MainLoop(
        training_algorithm,
        # Datastream used for training
        make_datastream(data.train, data.train_ind,
                        p.batch_size,
                        n_labeled=p.labeled_samples,
                        n_unlabeled=p.unlabeled_samples,
                        whiten=whiten,
                        cnorm=cnorm),
        model=Model(ladder.costs.total),
        extensions=[
            FinishAfter(after_n_epochs=p.num_epochs),

            # This will estimate the validation error using
            # running average estimates of the batch normalization
            # parameters, mean and variance
            ApproxTestMonitoring(
                [ladder.costs.class_clean, ladder.error.clean]
                + ladder.costs.denois.values(),
                make_datastream(data.valid, data.valid_ind,
                                p.valid_batch_size, whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                prefix="valid_approx"),

            # This Monitor is slower, but more accurate since it will first
            # estimate batch normalization parameters from training data and
            # then do another pass to calculate the validation error.
            FinalTestMonitoring(
                [ladder.costs.class_clean, ladder.error.clean]
                + ladder.costs.denois.values(),
                make_datastream(data.train, data.train_ind,
                                p.batch_size,
                                n_labeled=p.labeled_samples,
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                make_datastream(data.valid, data.valid_ind,
                                p.valid_batch_size,
                                n_labeled=len(data.valid_ind),
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                prefix="valid_final",
                after_n_epochs=p.num_epochs),

            TrainingDataMonitoring(
                [ladder.costs.total, ladder.costs.class_corr,
                 training_algorithm.total_gradient_norm]
                + ladder.costs.denois.values(),
                prefix="train", after_epoch=True),

            SaveParams(None, all_params, p.save_dir, after_epoch=True),
            SaveExpParams(p, p.save_dir, before_training=True),
            SaveLog(p.save_dir, after_training=True),
            ShortPrinting(short_prints),
            LRDecay(ladder.lr, p.num_epochs * p.lrate_decay, p.num_epochs,
                    after_epoch=True),
        ])
    main_loop.run()

if __name__ == "__main__":

    # 配置记录器
    logging.basicConfig(level=logging.INFO)

    # 这里定义了一堆后面要用到的lambda表达式，基本上都是用于字符转换的
    # 将字符串中的'-'转换为','
    rep = lambda s: s.replace('-', ',')

    # 切割字符串，分割符为','
    chop = lambda s: s.split(',')

    # 转换为int
    to_int = lambda ss: [int(s) for s in ss if s.isdigit()]

    # 转换为float
    to_float = lambda ss: [float(s) for s in ss]

    # 转换为布尔值
    def to_bool(s):
        if s.lower() in ['true', 't']:
            return True
        elif s.lower() in ['false', 'f']:
            return False
        else:
            raise Exception("Unknown bool value %s" % s)

    # 参数的单星号代表传入的是一个tuple，有多个一元值。
    # 如果是两个信号这代表传入的是一个dict，有多个二元值（值对）。
    # funs中有三个值，分别带入f、g、x，然后运算f(g(x))。
    def compose(*funs):
        return functools.reduce(lambda f, g: lambda x: f(g(x)), funs)

    # 不知道是啥
    # Functional parsing logic to allow flexible function compositions
    # as actions for ArgumentParser
    def funcs(additional_arg):
        class CustomAction(Action):
            def __call__(self, parser, args, values, option_string=None):
                def process(arg, func_list):
                    if arg is None:
                        return None
                    elif type(arg) is list:
                        return map(compose(*func_list), arg)
                    else:
                        return compose(*func_list)(arg)
                setattr(args, self.dest, process(values, additional_arg))
        return CustomAction

    # 命令行参数
    def add_train_params(parser, use_defaults):
        a = parser.add_argument
        default = lambda x: x if use_defaults else None
        # nargs="?" means we don't know how many parameters been requested
        # General hyper parameters and settings
        # 状态和结果的保存位置
        a("save_to", help="Destination to save the state and results",
          default=default("noname"), nargs="?")
        a("--num-epochs", help="Number of training epochs",
          type=int, default=default(150))
        # nargs="+" means must give at least one parameter
        a("--seed", help="Seed",
          type=int, default=default([1]), nargs='+')
        a("--dseed", help="Data permutation seed, defaults to 'seed'",
          type=int, default=default([None]), nargs='+')
        a("--labeled-samples", help="How many supervised samples are used",
          type=int, default=default(None), nargs='+')
        a("--unlabeled-samples", help="How many unsupervised samples are used",
          type=int, default=default(None), nargs='+')
        a("--dataset", type=str, default=default(['mnist']), nargs='+',
          choices=['mnist', 'cifar10'], help="Which dataset to use")
        a("--lr", help="Initial learning rate",
          type=float, default=default([0.002]), nargs='+')
        a("--lrate-decay", help="When to linearly start decaying lrate (0-1)",
          type=float, default=default([0.67]), nargs='+')
        a("--batch-size", help="Minibatch size",
          type=int, default=default([100]), nargs='+')
        # validation
        a("--valid-batch-size", help="Minibatch size for validation data",
          type=int, default=default([100]), nargs='+')
        a("--valid-set-size", help="Number of examples in validation set",
          type=int, default=default([10000]), nargs='+')

        # Hyperparameters controlling supervised path
        a("--super-noise-std", help="Noise added to supervised learning path",
          type=float, default=default([0.3]), nargs='+')
        # unsupervised learning
        a("--f-local-noise-std", help="Noise added encoder path",
          type=str, default=default([0.3]), nargs='+',
          action=funcs([tuple, to_float, chop]))
        a("--act", nargs='+', type=str, action=funcs([tuple, chop, rep]),
          default=default(["relu"]), help="List of activation functions")
        a("--encoder-layers", help="List of layers for f",
          type=str, default=default(()), action=funcs([tuple, chop, rep]))

        # Hyperparameters controlling unsupervised training
        a("--denoising-cost-x", help="Weight of the denoising cost.",
          type=str, default=default([(0.,)]), nargs='+',
          action=funcs([tuple, to_float, chop]))
        a("--decoder-spec", help="List of decoding function types",
          type=str, default=default(['sig']), action=funcs([tuple, chop, rep]))
        a("--zestbn", type=str, default=default(['bugfix']), nargs='+',
          choices=['bugfix', 'no'], help="How to do zest bn")

        # Hyperparameters used for Cifar training
        a("--contrast-norm", help="Scale of contrast normalization (0=off)",
          type=int, default=default([0]), nargs='+')
        a("--top-c", help="Have c at softmax?", action=funcs([to_bool]),
          default=default([True]), nargs='+')
        a("--whiten-zca", help="Whether to whiten the data with ZCA",
          type=int, default=default([0]), nargs='+')

    # UPD every parameter's get theirself default value. this will rewrite first parameter of method
    # ap = ArgumentParser("Semi-supervised experiment")
    ap = ArgumentParser(description="Semi-supervised experiment")

    # create a sub parser
    subparsers = ap.add_subparsers(dest='cmd', help='sub-command help')

    # TRAIN
    train_cmd = subparsers.add_parser('train', help='Train a new model')
    add_train_params(train_cmd, use_defaults=True)

    # EVALUATE
    load_cmd = subparsers.add_parser('evaluate', help='Evaluate test error')
    load_cmd.add_argument('load_from', type=str,
                          help="Destination to load the state from")
    load_cmd.add_argument('--data-type', type=str, default='test',
                          help="Data set to evaluate on")

    # 编译命令行参数
    args = ap.parse_args()

    # 运行'git rev-parse HEAD'，需要.git文件
    subp = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = subp.communicate()
    args.commit = out.strip()
    if err.strip():
        logger.error('Subprocess returned %s' % err.strip())

    # 计时器开始
    t_start = time.time()

    # 进入训练流程
    if args.cmd == "train":
        # 参数是数组的放在这个dict里
        listdicts = {k: v for k, v in vars(args).iteritems() if type(v) is list}

        # 参数不是数组的放在这里
        therest = {k: v for k, v in vars(args).iteritems() if type(v) is not list}

        # 打印参数 START
        # for k in listdicts:
        #     print('parameter:' + k + ' | value:' + str(listdicts[k]))
        # 打印参数 END

        # 感觉是去掉了单个值的伪装，即把只有一个值的tuple打开，去掉tuple这层壳
        # 逻辑有点复杂，可以简单点处理
        gen1, gen2 = tee(product(*listdicts.itervalues()))

        l = len(list(gen1))
        for i, d in enumerate(dict(izip(listdicts, x)) for x in gen2):
            if l > 1:
                logger.info('Training configuration %d / %d' % (i+1, l))

            # 更新therest中的d
            d.update(therest)

            # 训练开始
            if train(d) is None:
                break

    # 计时器结束
    logger.info('Took %.1f minutes' % ((time.time() - t_start) / 60.))
