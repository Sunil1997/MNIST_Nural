from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlretrieve

import numpy as np
import tensorflow as tf


#DataSets

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def maybe_download_iris_data(file_name, download_url):
  if not os.path.exists(file_name):
    print("Start Downloading>>>>>")
    urlretrieve(download_url, file_name)
    print("<<<<<Download")

  with open(file_name, 'r') as f:
    first_line = f.readline()
  num_elements = first_line.split(',')[0]
  return int(num_elements)

def input_fn(file_name, num_data, batch_size, is_training):

  def _parse_csv(rows_string_tensor):
    """Takes the string input tensor and returns tuple of (features, labels)."""
    # Last dim is the label.
    num_features = len(FEATURE_KEYS)
    num_columns = num_features + 1
    columns = tf.decode_csv(rows_string_tensor,
                            record_defaults=[[]] * num_columns)
    features = dict(zip(FEATURE_KEYS, columns[:num_features]))
    labels = tf.cast(columns[num_features], tf.int32)
    return features, labels

  def _input_fn():
    """The input_fn."""
    dataset = tf.data.TextLineDataset([file_name])
    # Skip the first line (which does not have data).
    dataset = dataset.skip(1)
    dataset = dataset.map(_parse_csv)

    if is_training:
      # For this small dataset, which can fit into memory, to achieve true
      # randomness, the shuffle buffer size is set as the total number of
      # elements in the dataset.
      dataset = dataset.shuffle(num_data)
      dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
  return _input_fn

def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  num_test_data = maybe_download_iris_data(IRIS_TEST, IRIS_TEST_URL)
  # print("data2:", num_test_data)

  num_training_data = maybe_download_iris_data(IRIS_TRAINING, IRIS_TRAINING_URL)
  # print("data1:",num_training_data)

  feature_columns = [
    tf.feature_column.numeric_column(key, shape=1) for key in FEATURE_KEYS]
  classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,model_dir="/home/sunil/log_model")

  train_input_fn = input_fn(IRIS_TRAINING, num_training_data, batch_size=32,
                            is_training=True)
  classifier.train(input_fn=train_input_fn, steps=400)

  # Eval.
  test_input_fn = input_fn(IRIS_TEST, num_test_data, batch_size=32,
                           is_training=False)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  main()