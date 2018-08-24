from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import gzip
import os
import csv
# from scipy.misc import imsave

from six.moves.urllib.request import urlretrieve

import tensorflow as tf
import numpy as np


SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def maybe_download(filename):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        print("Start Downloading>>>>>")
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


# def process_data(data,labels, file):
#     with open(file, 'wb') as csvFile:
#         writer = csv.writer(csvFile, delimiter=',', quotechar='"')
#         for i in range(len(data)):
#             imsave("mnist/train-images/" + str(i) + ".jpg", data[i][:, :, 0])
#             writer.writerow(["train-images/" + str(i) + ".jpg", labels[i]])


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    print(train_data_filename)
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    if not os.path.isdir("mnist/train-images"):
        os.makedirs("mnist/train-images")

    if not os.path.isdir("mnist/test-images"):
        os.makedirs("mnist/test-images")





    # test_file = "mnist/test-labels.csv"
    # train_file = "mnist/train-labels.csv"
    #
    # process_data(train_data,train_labels,train_file)
    # process_data(test_data,test_labels,test_file)


if __name__ == '__main__':
    main()