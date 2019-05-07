from __future__ import absolute_import, division

import bisect
import gzip
import hashlib
import io
import os

import apache_beam as beam
import numpy as np
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions
from logzero import logger
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.platform import gfile

tf.enable_eager_execution()

TFRECORD_OUTFILE = 'mnist'

FEATURE_DESCRIPTION = {
    'height': tf.FixedLenFeature([], tf.int64, default_value=0),
    'width': tf.FixedLenFeature([], tf.int64, default_value=0),
    'depth': tf.FixedLenFeature([], tf.int64, default_value=0),
    'label': tf.FixedLenFeature([], tf.int64, default_value=0),
    'image_raw': tf.FixedLenFeature([], tf.string, default_value=''),
}

SLICE = slice(0, 10)
SPLITS = ['train', 'eval']


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
        f: A file object that can be passed into a gzip reader.

    Returns:
        data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
        ValueError: If the bytestream does not start with 2051.

    """
    logger.info('Extracting: %s', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.

    Returns:
        labels: a 1D uint8 numpy array.

    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    logger.info('Extracting: %s', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return tf.one_hot(labels, num_classes)
        return labels


def get_images_and_labels(images_path, labels_path):
    """
    Extract gzip images/labels from path
    """
    with gfile.Open(images_path, 'rb') as f:
        images = extract_images(f)

    with gfile.Open(labels_path, 'rb') as f:
        labels = extract_labels(f)

    logger.info('images shape: %s', images.shape)
    logger.info('labels shape: %s', labels.shape)

    return images, labels


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_value(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_images_and_labels_w_index(images, labels):
    """
    Attach indexes to each record to be used for `beam.CoGroupByKey`

    TODO: also to be used by `beam._partition_fn` in the future
    """
    images_w_index = [(i, x) for i, x in enumerate(images)]
    labels_w_index = [(i, x) for i, x in enumerate(labels)]
    return images_w_index, labels_w_index


def group_by_tf_example(key_value):
    _, value = key_value
    image = value['image'][0]
    label = value['label'][0]
    height, width, depth = image.shape
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(label)),
            'image_raw': _int64_feature_value(
              np.reshape(image, (height*width*depth)))
        }))
    return example


def _partition_fn(
        record,
        num_partitions,  # pylint: disable=unused-argument
        buckets):
    bucket = int(hashlib.sha256(record).hexdigest(), 16) % buckets[-1]
    # For example, if buckets is [10,50,80], there will be 3 splits:
    #   bucket >=0 && < 10, returns 0
    #   bucket >=10 && < 50, returns 1
    #   bucket >=50 && < 80, returns 2
    int_split = bisect.bisect(buckets, bucket)
    return int_split


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _ImageToExample(pipeline, input_dict):
    data_dir = input_dict['input-base']

    images_path = os.path.join(data_dir, 'images.gz')
    labels_path = os.path.join(data_dir, 'labels.gz')

    images, labels = get_images_and_labels(images_path, labels_path)

    images_w_index, labels_w_index = get_images_and_labels_w_index(
        images, labels)

    # Beam Pipeline
    image_line = pipeline | "CreateImage" >> beam.Create(images_w_index[SLICE])
    label_line = pipeline | "CreateLabel" >> beam.Create(labels_w_index[SLICE])
    group_by = ({
        'label': label_line,
        'image': image_line
    }) | beam.CoGroupByKey()
    return (group_by | "GroupByToTfExample" >> beam.Map(group_by_tf_example))


def write_tfrecords():
    """
    Main write function
    """
    maybe_download()
    input_dict = {'input-base': '/tmp/data/mnist/val/'}
    buckets = [50, 100]
    with beam.Pipeline(options=PipelineOptions()) as p:
        tf_example = p | "InputSourceToExample" >> _ImageToExample(input_dict)

        serialize = (
            tf_example | 'SerializeDeterministically' >>
            beam.Map(lambda x: x.SerializeToString(deterministic=True)))

        example_splits = (serialize | 'SplitData' >> beam.Partition(
            _partition_fn, len(buckets), buckets))

        for i, example_split in enumerate(example_splits):
            split_name = SPLITS[i]
            (example_split | "Write." + split_name >> beam.io.WriteToTFRecord(
                split_name + '-' + TFRECORD_OUTFILE, file_name_suffix='.gz'))


def maybe_download():
    """
    Will download MNIST gzip files and move them to the correct
    locations if not already present
    """
    should_download = False
    data_dir = '/tmp/data/mnist'

    for x in ['train', 'val']:
        train_or_val_dir = os.path.join(data_dir, x)
        if not os.path.exists(train_or_val_dir):
            os.makedirs(train_or_val_dir)

        for fname in ['images', 'labels']:
            idx = 3 if x == 'images' else 1
            if x == 'train':
                path_to_check = os.path.join(
                    train_or_val_dir, 'train-{}-idx1-ubyte.gz'.format(idx, fname))
            else:
                path_to_check = os.path.join(
                    train_or_val_dir, 't10k-{}-idx1-ubyte.gz'.format(fname))

            if not os.path.isfile(path_to_check):
                should_download = True

    if not should_download:
        logger.info('data already present, no need to download')
        return

    # downloads MNIST datasets
    data_sets = mnist.read_data_sets(data_dir)

    # move to desired locations and rename
    def move_to_dest(target_dir, from_file, to_file):
        try:
            os.rename(os.path.join(data_dir, from_file),
                      os.path.join(target_dir, to_file))
        except OSError:
            # file already moved
            pass

    # train
    train_dir = os.path.join(data_dir, 'train')
    move_to_dest(train_dir, 'train-images-idx3-ubyte.gz', 'images.gz')
    move_to_dest(train_dir, 'train-labels-idx1-ubyte.gz', 'labels.gz')

    # test
    val_dir = os.path.join(data_dir, 'val')
    move_to_dest(val_dir, 't10k-images-idx3-ubyte.gz', 'images.gz')
    move_to_dest(val_dir, 't10k-labels-idx1-ubyte.gz', 'labels.gz')


def get_raw_dataset(filename):
    filenames = [filename]
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_record(dataset, idx=0):
    for i, x in enumerate(dataset.take(idx + 1)):
        if i == idx:
            return x


def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, FEATURE_DESCRIPTION)


def convert_parsed_record_to_ndarray(parsed_record):
    x = parsed_record['image_raw']
    x_np = x.numpy()
    bytestream = io.BytesIO(x_np)
    rows = 28
    cols = 28
    num_images = 1
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    shape = (rows, cols, num_images)
    data = data.reshape(*shape)
    assert isinstance(data, np.ndarray), type(data)
    assert data.shape == shape
    return data


def read_tfrecord(
        tfrecord_infile='{}-00000-of-00001.gz'.format(TFRECORD_OUTFILE),
        idx=0):
    """
    Main read function

    Reads a single image TFRecord and returns it as a np.ndarray
    """
    raw_dataset = get_raw_dataset(tfrecord_infile)

    parsed_dataset = raw_dataset.map(_parse_function)

    parsed_record = get_record(parsed_dataset, idx)

    return convert_parsed_record_to_ndarray(parsed_record)
