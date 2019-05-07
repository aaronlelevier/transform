# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example of sentiment analysis using IMDB movie review dataset.

But, it's an MNIST Example
"""

# pylint: disable=g-bad-import-order
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import pprint
import tempfile

import apache_beam as beam
import tensorflow as tf
from logzero import logger

import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from examples import aaron_rw_tfrecord as arw
from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema

# GOOGLE-INITIALIZATION




VOCAB_SIZE = 20000
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_EPOCHS = 2
NUM_TRAIN_INSTANCES = 60000
NUM_TEST_INSTANCES = 10000

# MNIST
HEIGHT = 28
WIDTH = 28
DEPTH = 1

# REVIEW_KEY = 'review'
# REVIEW_WEIGHT_KEY = 'review_weight'
# LABEL_KEY = 'label'
HEIGHT_KEY = 'height'
WIDTH_KEY = 'width'
DEPTH_KEY = 'depth'
LABEL_KEY = 'label'
IMAGE_RAW_KEY = 'image_raw'

# RAW_DATA_FEATURE_SPEC = {
#     REVIEW_KEY: tf.io.FixedLenFeature([], tf.string),
#     LABEL_KEY: tf.io.FixedLenFeature([], tf.int64)
# }
RAW_DATA_FEATURE_SPEC = { # known as the FEATURE_DESCRIPTION in TFX
    HEIGHT_KEY: tf.FixedLenFeature([], tf.int64),
    WIDTH_KEY: tf.FixedLenFeature([], tf.int64),
    DEPTH_KEY: tf.FixedLenFeature([], tf.int64),
    LABEL_KEY: tf.FixedLenFeature([], tf.int64),
    IMAGE_RAW_KEY: tf.FixedLenFeature([], tf.string),
}

RAW_DATA_METADATA = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec(RAW_DATA_FEATURE_SPEC))

DELIMITERS = '.,!?() '

# Names of temp files
SHUFFLED_TRAIN_DATA_FILEBASE = 'train_shuffled'
SHUFFLED_TEST_DATA_FILEBASE = 'test_shuffled'
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'

# Functions for preprocessing

MNIST_DATA_DIR = '/tmp/data/mnist/'


@beam.ptransform_fn
def ReadAndShuffleData(pcoll, train_or_val_str):
  """
  pcoll is a PCollection of incoming `tft.coders.ExampleProtoCoder`
  """
  data_dir = os.path.join(MNIST_DATA_DIR, train_or_val_str)

  images_path = os.path.join(data_dir, 'images.gz')
  labels_path = os.path.join(data_dir, 'labels.gz')

  images, labels = arw.get_images_and_labels(images_path, labels_path)

  images_w_index, labels_w_index = arw.get_images_and_labels_w_index(
      images, labels)

  # Beam Pipeline
  image_line = pcoll | "CreateImage" >> beam.Create(images_w_index[arw.SLICE])
  label_line = pcoll | "CreateLabel" >> beam.Create(labels_w_index[arw.SLICE])
  group_by = ({'label': label_line, 'image': image_line}) | beam.CoGroupByKey()

  # combines images and labels into a single TFRecord
  all_examples = (group_by
                  | "GroupByToTfExample" >> beam.Map(group_by_dict))

  # shuffles TFRecords
  return (all_examples | 'Shuffle' >> Shuffle())


def group_by_dict(key_value):
  # first value - is the index used by `beam.CoGroupByKey` to join the
  #   image/label from diff files, which we no longer need, so ignore it
  _, value = key_value
  image = value['image'][0]
  label = value['label'][0]
  height, width, depth = image.shape
  return {
    HEIGHT_KEY: HEIGHT,
    WIDTH_KEY: WIDTH,
    DEPTH_KEY: DEPTH,
    LABEL_KEY: label,
    IMAGE_RAW_KEY: image.tostring()
  }


# pylint: disable=invalid-name
@beam.ptransform_fn
def Shuffle(pcoll):
  """Shuffles a PCollection.  Collection should not contain duplicates."""
  return (
      pcoll
      | 'PairWithHash' >> beam.Map(lambda x: (hash(x[IMAGE_RAW_KEY]), x))
      | 'GroupByHash' >> beam.GroupByKey()
      | 'DropHash' >> beam.FlatMap(lambda hash_and_values: hash_and_values[1]))


def read_and_shuffle_data(train_neg_filepattern, train_pos_filepattern,
                          test_neg_filepattern, test_pos_filepattern,
                          working_dir):
  """Read and shuffle the data and write out as a TFRecord of Example protos.

  Read in the data from the positive and negative examples on disk, shuffle it
  and write it out in TFRecord format.
  transform it using a preprocessing pipeline that removes punctuation,
  tokenizes and maps tokens to int64 values indices.

  Args:
    train_neg_filepattern: Filepattern for training data negative examples
    train_pos_filepattern: Filepattern for training data positive examples
    test_neg_filepattern: Filepattern for test data negative examples
    test_pos_filepattern: Filepattern for test data positive examples
    working_dir: Directory to write shuffled data to
  """
  with beam.Pipeline() as pipeline:
    coder = tft.coders.ExampleProtoCoder(RAW_DATA_METADATA.schema)

    # train - shuffle data step
    _ = (pipeline
         | 'ReadAndShuffleTrain' >> ReadAndShuffleData('train')
         | 'EncodeTrainData' >> beam.Map(coder.encode)
         | 'WriteTrainData' >> beam.io.WriteToTFRecord(os.path.join(
             working_dir, SHUFFLED_TRAIN_DATA_FILEBASE), file_name_suffix='.gz'))

    # # test - shuffle data step
    _ = (pipeline
         | 'ReadAndShuffleTest' >> ReadAndShuffleData('val')
         | 'EncodeTestData' >> beam.Map(coder.encode)
         | 'WriteTestData' >> beam.io.WriteToTFRecord(
             os.path.join(working_dir, SHUFFLED_TEST_DATA_FILEBASE), file_name_suffix='.gz'))
    # # pylint: enable=no-value-for-parameter


def transform_data(working_dir):
  """Transform the data and write out as a TFRecord of Example protos.

  Read in the data from the positive and negative examples on disk, and
  transform it using a preprocessing pipeline that removes punctuation,
  tokenizes and maps tokens to int64 values indices.

  Args:
    working_dir: Directory to read shuffled data from and write transformed data
        and metadata to.
  """

  with beam.Pipeline() as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      coder = tft.coders.ExampleProtoCoder(RAW_DATA_METADATA.schema)
      train_data = (
          pipeline
          | 'ReadTrain' >> beam.io.ReadFromTFRecord(
              os.path.join(working_dir, SHUFFLED_TRAIN_DATA_FILEBASE + '*'))
          | 'DecodeTrain' >> beam.Map(coder.decode))

      test_data = (
          pipeline
          | 'ReadTest' >> beam.io.ReadFromTFRecord(
              os.path.join(working_dir, SHUFFLED_TEST_DATA_FILEBASE + '*'))
          | 'DecodeTest' >> beam.Map(coder.decode))

      def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        return inputs
        # review = inputs[REVIEW_KEY]
        # # Here tf.compat.v1.string_split behaves differently from
        # # tf.strings.split.
        # review_tokens = tf.compat.v1.string_split(review, DELIMITERS)
        # review_indices = tft.compute_and_apply_vocabulary(review_tokens,
        #                                                   top_k=VOCAB_SIZE)
        # # Add one for the oov bucket created by compute_and_apply_vocabulary.
        # review_bow_indices, review_weight = tft.tfidf(review_indices,
        #                                               VOCAB_SIZE + 1)
        # return {
        #     REVIEW_KEY: review_bow_indices,
        #     REVIEW_WEIGHT_KEY: review_weight,
        #     LABEL_KEY: inputs[LABEL_KEY]
        # }

      (transformed_train_data, transformed_metadata), transform_fn = (
          (train_data, RAW_DATA_METADATA)
          | 'AnalyzeAndTransform' >>
          tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data_coder = tft.coders.ExampleProtoCoder(
          transformed_metadata.schema)

      transformed_test_data, _ = ((
          (test_data, RAW_DATA_METADATA), transform_fn)
                                  | 'Transform' >> tft_beam.TransformDataset())

      _ = (transformed_train_data
           | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
           | 'WriteTrainData' >> beam.io.WriteToTFRecord(
               os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE), file_name_suffix='.gz'))

      _ = (transformed_test_data
           | 'EncodeTestData' >> beam.Map(transformed_data_coder.encode)
           | 'WriteTestData' >> beam.io.WriteToTFRecord(
               os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE), file_name_suffix='.gz'))

      # Will write a SavedModel and metadata to two subdirectories of
      # working_dir, given by tft.TRANSFORM_FN_DIR and
      # tft.TRANSFORMED_METADATA_DIR respectively.
      _ = (transform_fn
           | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))


# Functions for training


def _make_training_input_fn(tf_transform_output, transformed_examples,
                            batch_size):
  """Creates an input function reading from transformed data.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    transformed_examples: Base filename of examples.
    batch_size: Batch size.

  Returns:
    The input function for training or eval.
  """

  def input_fn():
    """Input function for training and eval."""
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=transformed_examples,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset,
        shuffle=True)

    transformed_features = tf.compat.v1.data.make_one_shot_iterator(
        dataset).get_next()

    # Extract features and label from the transformed tensors.
    # TODO(b/30367437): make transformed_labels a dict.
    transformed_labels = transformed_features.pop(LABEL_KEY)

    return transformed_features, transformed_labels

  return input_fn


def _make_serving_input_fn(tf_transform_output):
  """Creates an input function reading from raw data.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.

  Returns:
    The serving input function.
  """
  raw_feature_spec = RAW_DATA_METADATA.schema.as_feature_spec()
  # Remove label since it is not available during serving.
  raw_feature_spec.pop(LABEL_KEY)

  def serving_input_fn():
    """Input function for serving."""
    # Get raw features by generating the basic serving input_fn and calling it.
    # Here we generate an input_fn that expects a parsed Example proto to be fed
    # to the model at serving time.  See also
    # tf.estimator.export.build_raw_serving_input_receiver_fn.
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    # Apply the transform function that was used to generate the materialized
    # data.
    raw_features = serving_input_receiver.features
    transformed_features = tf_transform_output.transform_raw_features(
        raw_features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)

  return serving_input_fn


def get_feature_columns(tf_transform_output):
  """Returns the FeatureColumns for the model.

  Args:
    tf_transform_output: A `TFTransformOutput` object.

  Returns:
    A list of FeatureColumns.
  """
  del tf_transform_output  # unused
  # Unrecognized tokens are represented by -1, but
  # categorical_column_with_identity uses the mod operator to map integers
  # to the range [0, bucket_size).  By choosing bucket_size=VOCAB_SIZE + 1, we
  # represent unrecognized tokens as VOCAB_SIZE.
  review_column = tf.feature_column.categorical_column_with_identity(
      REVIEW_KEY, num_buckets=VOCAB_SIZE + 1)
  weighted_reviews = tf.feature_column.weighted_categorical_column(
      review_column, REVIEW_WEIGHT_KEY)

  return [weighted_reviews]


def train_and_evaluate(working_dir,
                       num_train_instances=NUM_TRAIN_INSTANCES,
                       num_test_instances=NUM_TEST_INSTANCES):
  """Train the model on training data and evaluate on evaluation data.

  Args:
    working_dir: Directory to read transformed data and metadata from.
    num_train_instances: Number of instances in train set
    num_test_instances: Number of instances in test set

  Returns:
    The results from the estimator's 'evaluate' method
  """
  tf_transform_output = tft.TFTransformOutput(working_dir)

  run_config = tf.estimator.RunConfig()

  estimator = tf.estimator.LinearClassifier(
      feature_columns=get_feature_columns(tf_transform_output),
      config=run_config,
      loss_reduction=tf.compat.v1.losses.Reduction.SUM)

  # Fit the model using the default optimizer.
  train_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + '*'),
      batch_size=TRAIN_BATCH_SIZE)
  estimator.train(input_fn=train_input_fn,
                  max_steps=TRAIN_NUM_EPOCHS * num_train_instances /
                  TRAIN_BATCH_SIZE)

  # Evaluate model on eval dataset.
  eval_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE + '*'),
      batch_size=1)
  result = estimator.evaluate(input_fn=eval_input_fn, steps=num_test_instances)

  # Export the model.
  serving_input_fn = _make_serving_input_fn(tf_transform_output)
  exported_model_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
  estimator.export_savedmodel(exported_model_dir, serving_input_fn)

  return result


def get_filepatterns():
  paths = []
  for dir_name in ['train', 'val']:
    base_dir_name = os.path.join('/tmp/data/mnist')
    train_or_val_dir = os.path.join(base_dir_name, dir_name)
    if not os.path.exists(train_or_val_dir):
      os.makedirs(train_or_val_dir)

    for fname in ['images', 'labels']:
      paths.append(os.path.join(train_or_val_dir, '{}.gz'.format(fname)))

  assert len(paths) == 4

  return paths


def main():
  parser = argparse.ArgumentParser()

  # input_data_dir - /tmp/data/mnist
  parser.add_argument('input_data_dir',
                      help='path to directory containing input data')

  # working_dir - is an optional arg, or it defaults to the "input_data_dir"
  parser.add_argument('--working_dir',
                      help='path to directory to hold transformed data')
  args = parser.parse_args()

  if args.working_dir:
    working_dir = args.working_dir
  else:
    working_dir = tempfile.mkdtemp(dir=args.input_data_dir)

  filepatterns = get_filepatterns()

  read_and_shuffle_data(*filepatterns, working_dir=working_dir)

  # TODO(aaronlelevier): implement after read/shuffle works
  transform_data(working_dir)
  # results = train_and_evaluate(working_dir)

  pprint.pprint(results)


if __name__ == '__main__':
  main()
