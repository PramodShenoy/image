# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 13:47:02 2018

@author: Pramod
"""
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

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf_spiky(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
'''
with tf.Session() as sess:

    x = tf.constant([0.2,0.7,1.2,1.7])
    y = tf_spiky(x)
    tf.initialize_all_variables().run()

    print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())
    '''

### Placeholders

x = tf.placeholder(tf.float32,shape=[None,784])

y_true = tf.placeholder(tf.float32,shape=[None,10])

### Layers

x_image = tf.reshape(x,[-1,28,28,1])
print(x_image.get_shape())
# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can change the 32 output, that essentially represents the amount of filters used
# You need to pass in 32 to the next input though, the 1 comes from the original input of 
# a single image.
convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
convo_1 = tf.reshape(convo_1,shape=[-1,28,28,32])
print(convo_1.get_shape())
convo_1_pooling = max_pool_2by2(convo_1)
print(convo_1_pooling.get_shape())
# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can actually change the 64 output if you want, you can think of that as a representation
# of the amount of 6by6 filters used.
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2 = tf.reshape(convo_2,shape=[-1,14,14,64])
print(convo_2.get_shape())
convo_2_pooling = max_pool_2by2(convo_2)
print(convo_2_pooling.get_shape()) 
# Why 7 by 7 image? Because we did 2 pooling layers, so (28/2)/2 = 7
# 64 then just comes from the output of the previous Convolution
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
print(convo_2_flat.get_shape())
#full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))
convo_2_flat=tf.reshape(convo_2_flat,shape=[tf.shape(convo_2_flat)[0],1024])
print(convo_2_flat.get_shape())
full_layer_one = tf_spiky(normal_full_layer(convo_2_flat,1024))
full_layer_one = tf.reshape(full_layer_one,shape=[-1,1024])
print(full_layer_one.get_shape())
# NOTE THE PLACEHOLDER HERE!
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
print(full_one_dropout.get_shape())
y_pred = normal_full_layer(full_layer_one,10)
print(y_pred.get_shape())
### Loss Function

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
print(cross_entropy.get_shape())
### Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)

### Intialize Variables

init = tf.global_variables_initializer()

### Session

steps = 5000

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        batch_x , batch_y = mnist.train.next_batch(50)
        print(np.shape(batch_x))
        print(np.shape(batch_y))
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')
