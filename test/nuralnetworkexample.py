from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import gzip
from IPython.display import display,Image
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import csv
from scipy.misc import imsave


SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
WORK_DIRECTORY = "Data"
LAST_PERCENTAGE_REPORTED = None
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
NUM_LABELS = 10

def download_progres_bar(count,blockSize,totalSize):
    global LAST_PERCENTAGE_REPORTED
    percent = int(count*blockSize*100/totalSize)

    if LAST_PERCENTAGE_REPORTED != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%"% percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        LAST_PERCENTAGE_REPORTED = percent

def maybe_dowmload(filename, expected_bytes):
    if not os.path.exists(WORK_DIRECTORY):
        os.makedirs(WORK_DIRECTORY)
    dest_filename = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(dest_filename):
        print('Download of ',filename,'Start')
        dest_filename, _ = urlretrieve(SOURCE_URL+filename, dest_filename, reporthook=download_progres_bar)
        print("Download Complete")
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and Verified',dest_filename)
    else:
        raise Exception(
            'failed to verify'+dest_filename
        )
    return dest_filename


def maybe_extract_data(filename, num_images):
    print('Extracting',filename)
    with gzip.open(filename) as bytestram:
        bytestram.read(16)
        buf = bytestram.read(IMAGE_SIZE*IMAGE_SIZE*num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

def maybe_extract_labels(filename,num_images):
    print('Extracting:', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buffer = bytestream.read(1 * num_images)
        labels = np.frombuffer(buffer,dtype=np.uint8).astype(np.int64)
        return labels


TRAINING_IMAGES_FILENAME = maybe_dowmload('train-images-idx3-ubyte.gz',9912422)
TRAINING_LABELS_FILENAME = maybe_dowmload('train-labels-idx1-ubyte.gz',28881)
TESTING_IMAGES_FILENAME  = maybe_dowmload('t10k-images-idx3-ubyte.gz',1648877)
TESTING_LABELS_FILENAME  = maybe_dowmload('t10k-labels-idx1-ubyte.gz',4542)

MNIST_TRAIN_IMAGES  = maybe_extract_data(TRAINING_IMAGES_FILENAME, 60000)
MNIST_TRAIN_LABELS = maybe_extract_labels(TRAINING_LABELS_FILENAME, 60000)
MNIST_TEST_IMAGES   = maybe_extract_data(TESTING_IMAGES_FILENAME, 10000)
MNIST_TEST_LABELS   = maybe_extract_labels(TESTING_LABELS_FILENAME, 10000)

# print(MNIST_TRAIN_LABELS)
if not os.path.isdir("mnist/train-images"):
    os.makedirs("mnist/train-images")

if not os.path.isdir("mnist/test-images"):
    os.makedirs("mnist/test-images")


with open ("mnist/train-labels.csv",'wb')as csvFile:
    writer = csv.writer(csvFile, delimiter=',',quotechar='"')
    for i in range(len(MNIST_TRAIN_IMAGES)):
        imsave("mnist/train-images/"+ str(i) + ".jpg", MNIST_TRAIN_IMAGES[i][:, :, 0])
        # writer.writerow(["train-images/" + str(i) + ".jpg", MNIST_TRAIN_LABELS[i]])

with open("mnist/test-labels.csv","wb") as csvFile:
    writer = csv.writer(csvFile,delimiter=',', quotechar='"')
    for i in range (len(MNIST_TRAIN_IMAGES)):
        imsave(b"mnist/test-images/" + str(i) + b".jpg", MNIST_TEST_IMAGES[i][:, :, 0])
        # writer.writerow([b"test-images/" + str(i) + b".jpg", MNIST_TEST_LABELS[i]])