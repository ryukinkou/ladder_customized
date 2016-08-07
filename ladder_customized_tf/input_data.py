# -*- coding: utf-8 -*-

"""
Functions for downloading and reading MNIST data.
Functions for loading tween data from local file system.
"""

from __future__ import print_function

import numpy
import matplotlib.image as mpimg

import gzip
import os
import urllib

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
TWEEN_DATA_DIR = './tween_data'
VALIDATION_SIZE = 0


def maybe_download(filename, work_directory):

    """Download the data from Yann's website, unless it's already here."""

    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    file_path = os.path.join(work_directory, filename)
    if not os.path.exists(file_path):
        file_path, _ = urllib.urlretrieve(SOURCE_URL + filename, file_path)
        stat_info = os.stat(file_path)
        print('Successfully downloaded', filename, stat_info.st_size, 'bytes.')
    return file_path


def _read32(bytestream):

    """Read 4 bytes in big endian order."""

    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):

    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""

    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' % (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def rgb2gray(rgb):

    """Convert rgb to gray-scale."""

    return numpy.dot(rgb[..., :3], [0.299 * 255, 0.587 * 255, 0.114 * 255])


def load_tween_images_and_labels(filename, one_hot=False):

    """
    load tween images and labels from file system
    keep the same output structure of extract_images/extract_labels function
    """

    print('Loading', filename)

    data_set_list = []
    label_list = []

    cols = 0
    rows = 0
    num_images = 0
    num_dirs = 0

    for root, dirs, files in os.walk(filename):

        dirs.sort()

        if num_dirs == 0:
            num_dirs = len(dirs)

        for f in sorted(files):

            sample_name = os.path.join(root, f)
            rgb = mpimg.imread(sample_name)
            gray_scale = rgb2gray(rgb)

            if cols == 0 or rows == 0 or num_images == 0:
                num_images = len(files) * num_dirs
                rows = len(gray_scale)
                cols = len(gray_scale[0])

            data_list = gray_scale.reshape(1, gray_scale.size).tolist()
            data_set_list.extend(data_list[0])

            label = int(root.replace(filename + '/', ''))
            label_list.append(label)

    output_data_list = numpy.array(data_set_list, dtype=numpy.uint8)
    output_data_list = output_data_list.reshape(num_images, rows, cols, 1)

    output_label_list = numpy.array(label_list, dtype=numpy.uint8)

    if one_hot:
        output_label_list = dense_to_one_hot(output_label_list)

    return output_data_list, output_label_list


def dense_to_one_hot(labels_dense, num_classes=10):

    """Convert class labels from scalars to one-hot vectors."""

    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):

    """Extract the labels into a 1D uint8 numpy array [index]."""

    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):

    def __init__(self, images, labels, fake_data=False):

        # fake data process
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], \
                ("images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows * columns] (assuming depth == 1).
        # Combine a big matrix [ num_examples , single_sample ] from entity data set.
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):

        """Return the next `batch_size` examples from this data set."""

        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


# 混合数据集
class SemiMNISTDataSet(object):

    def __init__(self, images, labels, num_labeled):

        # 设定标记的数量
        self.num_labeled = num_labeled

        # 未标记数据集设定
        # 使用全部数据集为未标记数据
        self.all_data_set = DataSet(images, labels)
        self.unlabeled_data_set = self.all_data_set

        # 未标记数据集的数量
        self.num_examples = self.unlabeled_data_set.num_examples

        # 生成索引数组并打散顺序
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)

        # 打散数据集
        images = images[shuffled_indices]
        labels = labels[shuffled_indices]

        # 随机挑选出num_labeled个数据作为标记数据
        y = numpy.array([numpy.arange(10)[l == 1][0] for l in labels])

        num_classes = y.max() + 1
        num_from_each_class = num_labeled / num_classes
        i_labeled = []
        for clazz in range(num_classes):

            # 先把y==clazz的index抽出来，再取其前num_from_each_class个
            i = indices[y == clazz][:num_from_each_class]
            i_labeled += list(i)

        l_images = images[i_labeled]
        l_labels = labels[i_labeled]

        self.labeled_data_set = DataSet(l_images, l_labels)

    def extend_labeled_data_set(self, extra_images, extra_labels):
        print(len(extra_images))
        print(len(extra_labels))

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_data_set.next_batch(batch_size)
        if batch_size > self.num_labeled:
            labeled_images, labels = self.labeled_data_set.next_batch(self.num_labeled)
        else:
            labeled_images, labels = self.labeled_data_set.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return images, labels


# 读取数据集处理
def read_data_sets(train_dir, num_labeled=100, fake_data=False, one_hot=False):

    class DataSets(object):
        pass

    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)

    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)

    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)

    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    # load
    tween_images, tween_labeled = load_tween_images_and_labels(TWEEN_DATA_DIR, True)

    data_sets.train = SemiMNISTDataSet(train_images, train_labels, num_labeled)

    data_sets.validation = DataSet(validation_images, validation_labels)

    data_sets.test = DataSet(test_images, test_labels)

    return data_sets


if __name__ == "__main__":

    # a = extract_images("./MNIST_data/train-images-idx3-ubyte.gz")
    # b, bl = load_tween_images_and_labels("./tween_data", True)
    # c = extract_labels("./MNIST_data/train-labels-idx1-ubyte.gz", True)
    read_data_sets("MNIST_data", num_labeled=100, one_hot=True)
