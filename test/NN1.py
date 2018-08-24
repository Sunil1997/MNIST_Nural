import numpy as np
import gzip
import math
IMAGE_SIZE = 28


def sigmoid(x):
    # return (2 / (1 + np.exp(-2 * (x)))) - 1
    # print(x,"=>",(np.exp(2*x)-1)/(np.exp(2*x) + 1))
    # return 1 / (1 + np.exp(-x))
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def derivation_sigmoid(x):
    # return 1 - (x ** 2)
    return x * (1.0 - x)


def softmax(x):
    # print np.sum(np.exp(x))
    e = np.exp(x-np.amax(x))
    # print(e)
    return e / (np.sum(e))


def extract_data(filename, num_images):
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data



def extract_labels(filename, num_images):
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def derivation_cross(x, y):
    return -1 * ((x * (1 / y)) + (1 - x) * (1 / (1 - y)))

def derivation_softmax(x):
    tem = sum(np.exp(x)) ** 2
    # print tem
    tem1 = sum(np.exp(x))
    a = []
    for i_ in range(len(x)):
        t1 = (np.exp(x[i_]) * (tem1 - np.exp(x[i_]))) / tem
        a.append(t1)
    return a

train_data_filename = "/home/sunil/PycharmProjects/test/Data/train-images-idx3-ubyte.gz"
train_data = extract_data(train_data_filename, 60000)
# print train_data
train_data_label = "/home/sunil/PycharmProjects/test/Data/train-labels-idx1-ubyte.gz"
tarin_data_label_1 = extract_labels(train_data_label, 60000)
input_array = []
counter = 0



input_neurons = train_data.shape[1] * train_data.shape[1]   # Number of Feature
out_neurons = 10
hidden_layer_nurons = 15
lr = 0.1
bias_hidden = np.ones((15, 1))
bias_output = np.ones((10, 1))
wih = np.random.uniform(-1, 1, size=(hidden_layer_nurons,input_neurons))
who = np.random.uniform(-1, 1, size=(out_neurons,hidden_layer_nurons))
# print wih.shape
# print who.shape


def neural_network(wih, who, train_data, tarin_data_label_1):
    for e_ in range(0, 1):
        for da_ in range(0, 16):
            # da_ = int(np.random.uniform(1, 3))
            c_error = []
            m1 = np.matrix((train_data[da_]))
            m2 = np.reshape(m1, (784, 1))
            m2 = ((m2 * 1) / 255)
            # print(m2.shape)

            # print(wih.shape)
            # print (sigmoid(np.matmul(wih, m2)))
            hidden_sigmoid = sigmoid((np.dot(wih, m2)))

            output = (np.dot(who, hidden_sigmoid))
            # print(output)

            output_softmax = softmax(output)
            # print(output, "=>", output_softmax)

            index = tarin_data_label_1[da_]
            for i in range(0, 10):
                if i != index:
                    c_error.append([0])
                else:
                    c_error.append([1])
            # c_error = np.matrix(c_error)
            a1 = np.array(c_error)
            a2 = np.array(output_softmax)

            # This is for Hidden -> Output change weight
            cross_entropy_derivation = derivation_cross(a1, a2)
            # print (cross_entropy_derivation)
            softmax_derivation = derivation_softmax(output)
            # print(softmax_derivation)
            sigmoid_derivation = derivation_softmax(hidden_sigmoid)
            sigmoid_derivation = np.reshape(sigmoid_derivation, (15, 1))

            m = np.reshape(softmax_derivation, (10, 1))
            d_output = cross_entropy_derivation * m
            # print(d_output.shape)
            # print(hidden_sigmoid.shape)

            who -= (d_output.dot(hidden_sigmoid.T))
            # who -= np.matmul(d_output,hidden_sigmoid.T)
            print(who)
            # # print(who.shape)
            # # # this is for Hidden -> Input
            # #
            d_hidden = who.T.dot(d_output)
            # d_hidden = np.matmul(who.T,d_output)
            # print(d_hidden.shape)
            d1 =  d_hidden * sigmoid_derivation
            # print(d1.shape)
            # print(m2.shape)
            wih -= np.matmul(d1,m2.T)
            # print(wih.shape)
            # wih -= d1.dot(m2.T)
            # print who[0]
            test = np.argmax(output_softmax)
            print (test, index)




neural_network(wih, who, train_data, tarin_data_label_1)







