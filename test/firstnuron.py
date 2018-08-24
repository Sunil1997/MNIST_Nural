import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope("layer"):
        with tf.name_scope("WEIGHTS"):
            weights = tf.Variable(tf.random_normal([in_size,out_size]),name='w')
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope("inputs"):
            Wx_plus_b = tf.add(tf.matmul(inputs,weights), biases)

        if activation_function == None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs


#make up some real data

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# plt.scatter(x_data,y_data)
# plt.show()

#define placeholder for inputs to network
with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32, [None,1],name='x_input')
    ys = tf.placeholder(tf.float32, [None,1],name='y_input')

#add hindden layer

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

#add output layers

prediction = add_layer(l1, 10, 1, activation_function=None)

#error between prdiction and real
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#important step
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.initialize_all_variables()
sess.run(init)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    #training
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

    if i%50 == 0:
        #to see step improvoment
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        #plot prdiction

        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(1)
