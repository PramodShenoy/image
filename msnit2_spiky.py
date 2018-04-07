import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

def spiky(x):
    r = x%1
    if x<=0.5:
        return r
    else:
        return 0
np_spiky = np.vectorize(spiky)

def spiky_d(x):
    r=x%1
    if r<=0.5:
        return 1
    else:
        return 0
np_spiky_d = np.vectorize(spiky_d)
np_d_spiky_32 = lambda x: np_spiky_d(x).astype(np.float32)


def tf_d_spiky(x,name=None):
    with ops.op_scope([x], name, "d_spiky") as name:
        y = tf.py_func(np_d_spiky_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]
    
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def spikygrad(op, grad):
    x = op.inputs[0]

    n_gr = tf_d_spiky(x)
    return grad * n_gr  

np_spiky_32 = lambda x: np_spiky(x).astype(np.float32)

def tf_spiky(x, name=None):

    with ops.op_scope([x], name, "spiky") as name:
        y = py_func(np_spiky_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=spikygrad)  # <-- here's the call to the gradient
        return y[0]

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.0001
epochs = 10
batch_size = 50

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784 - this is the flattened image data that is drawn from 
# mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [None, 784])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, 28, 28, 1])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf_spiky(out_layer)
    print(out_layer.get_shape())
    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')
    return out_layer

layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
#layer1 = tf.reshape(layer1,shape=[-1,14,14,32])
print(layer1.get_shape())
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
#layer2 = tf.reshape(layer1,shape=[-1,7,7,64])
print(layer2.get_shape())

flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
print(flattened.get_shape())
# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
print(wd1.get_shape())
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
print(bd1.get_shape())
dense_layer1 = tf.matmul(flattened, wd1) + bd1
print(dense_layer1.get_shape())
dense_layer1 = tf_spiky(dense_layer1)
#dense_layer1 = tf.reshape(dense_layer1,shape=[-1,1000])
print(dense_layer1.get_shape())

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
print(wd2.get_shape())
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
print(bd2.get_shape())
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
print(dense_layer2.get_shape())
y_ = tf.nn.softmax(dense_layer2)
print(y_.get_shape())
print(dense_layer2.get_shape())
print(y.get_shape())
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
print((cross_entropy.get_shape()))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    print(total_batch)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], 
                            feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, 
                       feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))