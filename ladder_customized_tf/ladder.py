# -*- coding: utf-8 -*-

# 已知bug 必须先行引入这两个库，否则会报错
import numpy
import matplotlib.image as mpimg
import sys

import tensorflow as tf
from tensorflow.python import control_flow_ops
import math
import os
import csv
from tqdm import tqdm

# 读取数据
import input_data

# 设置gpu使用数量
tf.app.flags.DEFINE_integer("num_gpus", 2, "How many GPUs to use.")

# 每一层的神经元数量设定，为全链接层
# 第一层为输入层，28*28
# 最后一层为分类输出层，有10个分类
layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

# 层数，大L
L = len(layer_sizes) - 1

# 总样本数：0有15000个，1-9个有5000个。
num_unlabeled_samples = 60000
num_labeled_samples = 1000

num_examples = num_unlabeled_samples + num_labeled_samples

# 全样本扫描循环次数
# 样本数量庞大，通过设定mini batch的大小分批扫描，所有样本都扫描一次算一次全样本扫描
num_epochs = 150

# 类别数量
num_labeled = 10

# 冷启动的lr值
starter_learning_rate = 0.02

# 经历15次全样本扫描后，lr值开始衰减
# epoch after which to begin learning rate decay
decay_after = 15

# mini batch的大小
batch_size = 100

# ( 总样本数 / mini_batch = 一次全样本扫描所需要的批次 ) * 全样本扫描次数 = 总的循环次数
# number of loop iterations
num_iter = (num_examples / batch_size) * num_epochs

# 为输入值的张量分配一块内存区域，类型为float32，shape为一维数组，数组长度为输入层神经元个数
inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]))
# 为输出值的张量分配一块内存区域，大小暂时未定
outputs = tf.placeholder(tf.float32)

# 创建两个lambda，用于初始化偏置参数与权重
# 偏置项的值由传入参数矢量化确定
bi = lambda inits, size, name: tf.Variable(inits * tf.ones([size]), name=name)
# 权值由随机生成的正态分布确定
wi = lambda shape, name: tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

# 制作层依赖pair making
# shapes of linear layers
shapes = zip(layer_sizes[:-1], layer_sizes[1:])

# 编码器（有监督学习）权值 W
# 解码器（无监督学习）全职 V
# batch norn中用于重构变换过程的beta（可学习）
# batch norn中用于重构变换过程的gamma（可学习）
weights = {'W': [wi(s, "W") for s in shapes], # Encoder weights
           'V': [wi(s[::-1], "V") for s in shapes], # Decoder weights
           'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(L)], # batch normalization parameter to shift the normalized value
           'gamma': [bi(1.0, layer_sizes[l+1], "beta") for l in range(L)]} # batch normalization parameter to scale the normalized value

# 调整噪点影响程度的scale值
noise_std = 0.3

# 去噪用的cost的超参数，用于控制每一层的重要度（importance）
denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10] # hyperparameters that denote the importance of each layer

# 合并两个二维tensor，0代表行合并，1代表列合并
join = lambda l, u: tf.concat(0, [l, u])

# 切片处理
# 切出前batch_size个样本作为标记数据
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x

# 切出batch_size到结束个样本作为未标记数据
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x

# pair
split_lu = lambda x: (labeled(x), unlabeled(x))

# 留下一个bool位
is_training = tf.placeholder(tf.bool)

print(denoising_cost)

exit()

# 设置衰减value，用于维持参数的移动平均（moving average）
# decay是指新的一轮数据（l层数据）进入时，旧的数据（l-1层数的数据）权重降低的百分比
ewma = tf.train.ExponentialMovingAverage(decay=0.99)
# this list stores the updates to be made to average mean and variance
bn_assigns = []

# batch norm处理
def batch_normalization(batch, mean=None, var=None):
    if mean == None or var == None:
        # 计算batch的均值与方差
        mean, var = tf.nn.moments(batch, axes=[0])
        # 对batch中的每一个数值进行基于逆标准差的normalization处理
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

# average mean and variance of all layers
# 为每一层的running_mean和running_var分配空间并初始化
running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]
running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in layer_sizes[1:]]

# 更新batch normalization
def update_mean_var_and_batch_normalization(batch, l):

    # 一个batch更新一次，而非一个epoch
    "batch normalize + update average mean and variance of layer l"
    mean, var = tf.nn.moments(batch, axes=[0])
    # 设定上一层的running_mean为mean值
    assign_mean = running_mean[l-1].assign(mean)
    assign_var = running_var[l-1].assign(var)

    # 应用衰减
    bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))

    # 优先计算玩assign_mean和assign_var后在计算bn值。是一种强制控制计算先后顺序的方法
    with tf.control_dependencies([assign_mean, assign_var]):
        return (batch - mean) / tf.sqrt(var + 1e-10)

# 编码器
def encoder(inputs, noise_std):

    # 生成正太分布的随机噪点，乘以noise_std调整噪点的权重
    h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std

    # d用来储存 激活前的值 激活后的值 平均值 方差
    # to store the pre-activation, activation, mean and variance for each layer
    d = {}

    # 把数据集分别切到两个（标记，未标记）序列中
    # The data for labeled and unlabeled examples are stored separately
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}

    # 设定第0层的值
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)

    # 逐层迭代
    for l in range(1, L+1):

        # logic layer start at 1
        current_logic_layer = l
        # data layer start at 0
        current_data_layer = current_logic_layer - 1
        # next data layer
        next_data_layer = current_data_layer + 1

        print "Current Layer ", current_logic_layer, " : ", layer_sizes[current_data_layer], " -> to next layer : ", layer_sizes[next_data_layer]
        d['labeled']['h'][current_data_layer], d['unlabeled']['h'][current_data_layer] = split_lu(h)

        # matmul 矩阵乘法，激活之前的运算
        # pre-activation
        z_pre = tf.matmul(h, weights['W'][l-1])

        # 算完继续分开
        z_pre_l, z_pre_u = split_lu(z_pre) # split labeled and unlabeled examples

        # 计算非标记数据的均值与方差
        m, v = tf.nn.moments(z_pre_u, axes=[0])

        # bn算法训练流程
        def training_batch_norm():
            # 训练两组encoder，一个是加入噪点的，一个是不加入噪点的
            # 且batch normalization中标记数据和未标记数据分开处理
            if noise_std > 0:
                # 对标记数据和非标记数据分别进行batch_norm，然后合并
                z = join(batch_normalization(z_pre_l), batch_normalization(z_pre_u, m, v))
                # 加入噪点，生成一个与z_pre同样大小的向量，用随机数填充，然后乘以随机噪点权重
                z += tf.random_normal(tf.shape(z_pre)) * noise_std
            else:
                # Clean encoder
                # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                # 如果要训练干净的编码器，并不需要加入随机噪点
                z = join(update_mean_var_and_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
            return z

	#else:
        # 进入评估分支
        def eval_batch_norm():
            # Evaluation batch normalization
            # obtain average mean and variance and use it to normalize the batch
    	    mean = ewma.average(running_mean[l-1])
    	    var = ewma.average(running_var[l-1])
            z = batch_normalization(z_pre, mean, var)
            # Instead of the above statement, the use of the following 2 statements containing a typo 
            # consistently produces a 0.2% higher accuracy for unclear reasons.
            # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
            # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
            return z

        # perform batch normalization according to value of boolean "training" placeholder:
        # training是一个bool值，根据改值的设定，确定是进入训练还是评价流程
        z = control_flow_ops.cond(is_training, training_batch_norm, eval_batch_norm)

        # 如果是输出层（最后一层），应用softmax函数
        if l == L:
            # use softmax activation in output layer
            h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
        # 如果不是输出层，使用ReLU激活函数
        else:
            # use ReLU activation in hidden layers
            h = tf.nn.relu(z + weights["beta"][l-1])
        d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
        d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v # save mean and variance of unlabeled examples for decoding
    d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
    return h, d

print "=== Corrupted Encoder ==="
y_c, corr = encoder(inputs, noise_std)

print "=== Clean Encoder ==="
# 设置noise_std为0训练一个clean encoder
y, clean = encoder(inputs, 0.0)

print "=== Decoder ==="

# 定义高斯去噪器，输入z corr，输出去噪后的预测值
# 论文17页
def g_gauss(z_c, u, size):
    "gaussian denoising function proposed in the original paper"
    wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
    a1 = wi(0., 'a1')
    a2 = wi(1., 'a2')
    a3 = wi(0., 'a3')
    a4 = wi(0., 'a4')
    a5 = wi(0., 'a5')

    a6 = wi(0., 'a6')
    a7 = wi(1., 'a7')
    a8 = wi(0., 'a8')
    a9 = wi(0., 'a9')
    a10 = wi(0., 'a10')

    mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
    v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

    z_est = (z_c - mu) * v + mu
    return z_est

# Decoder
z_est = {}
d_cost = [] # to store the denoising cost of all layers

# 从第L层开始，迭代到第0层
for l in range(L, -1, -1):
    print "Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l]

    # 取出每一层激活后的值（最终结果）
    z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]

    # get 没有的时候返回缺省值
    m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
    if l == L:
        u = unlabeled(y_c)
    else:
        # ？？
        u = tf.matmul(z_est[l+1], weights['V'][l])
    u = batch_normalization(u)
    # z_est是根据z(l+1)的数据还原出来的z的预测值（去噪后）
    z_est[l] = g_gauss(z_c, u, layer_sizes[l])
    # batch norm处理
    z_est_bn = (z_est[l] - m) / v
    # append the cost of this layer to d_cost
    # 逐层计算cost：z_est_bn - z
    # reduce_sum 跨越维度的计算sum值
    # reduce_mean 跨越维度的计算均值

    # 方差 square(z_est_bn - z)
    # 维度加法 reduce_sum(方差,1)
    d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

# calculate total unsupervised cost by adding the denoising cost of all layers

#
u_cost = tf.add_n(d_cost)

y_N = labeled(y_c)

# 负对数概率之和
cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1)) # supervised cost

# cost 叠加
loss = cost + u_cost # total cost

# ground truth的cost
pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1)) # cost used for prediction

# tf.argmax 返回最大值的索引
# 正确预测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(outputs, 1)) # no of correct predictions

# 精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

learning_rate = tf.Variable(starter_learning_rate, trainable=False)

# 使用adam算法动态调整步长，基于最初设定的learning_rate
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print "===  Loading Data ==="
mnist = input_data.read_data_sets("MNIST_data", n_labeled = num_labeled, one_hot=True)

# num_labeled = num_labeled
# num_labeled = 1002

saver = tf.train.Saver()

print "===  Starting Session ==="
sess = tf.Session()

i_iter = 0

ckpt = tf.train.get_checkpoint_state('checkpoints/') # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    i_iter = (epoch_n+1) * (num_examples/batch_size)
    print "Restored Epoch ", epoch_n
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    init  = tf.initialize_all_variables()
    sess.run(init)

print "=== Training ==="
print "Initial Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, is_training: False}), "%"

for i in tqdm(range(i_iter, num_iter)):
    images, labels = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={inputs: images, outputs: labels, is_training: True})
    if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
        epoch_n = i/(num_examples/batch_size)
        if (epoch_n+1) >= decay_after:
            # decay learning rate
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (num_epochs - (epoch_n+1)) # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
        # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%"
	with open('train_log', 'ab') as train_log:
            # write test accuracy to file "train_log"
            train_log_w = csv.writer(train_log)
            log_i = [epoch_n] + sess.run([accuracy], feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, is_training: False})
            train_log_w.writerow(log_i)

print "Final Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs: mnist.test.labels, is_training: False}), "%"

sess.close()
