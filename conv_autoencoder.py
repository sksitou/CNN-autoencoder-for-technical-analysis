import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from libs.activations import lrelu
from libs.utils import corrupt
'''
feed stocks with batch
'''


# %%
def build_ae(input_size,
                input_shape,
                n_pooling,
                n_filters,
                filter_sizes=[4, 4, 4, 4],
                corruption=False):
    n_filters = [1]*(n_filters+1)

    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    if len(x.get_shape()) == 2:
        x_tensor = tf.reshape(
            #x, [-1, x_dim, x_dim, n_filters[0]])
            x, [-1, input_size, 1, n_filters[0]])
    elif len(x.get_shape()) == 5:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    if corruption:
        current_input = corrupt(current_input)

    # %%
    # Build the encoder
    def max_pool_1x4(x):
        return tf.nn.max_pool(x, ksize=[1,1,n_pooling,1], strides=[1,1,n_pooling,1], padding='SAME')
    def avg_pool_1x4(x):
        return tf.nn.avg_pool(x, ksize=[1,n_pooling,1,1], strides=[1,n_pooling,1,1], padding='VALID')
    def un_avg_pool_1x4(x):
        divide = 1.0/n_pooling
        y = tf.mul(x,divide)
        return tf.tile(y,tf.pack([1,n_pooling,1,1]))

    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        #n_input = current_input.get_shape().as_list()[3]
        n_input = current_input.get_shape().as_list()[3]
        
        #print str(layer_i) + '---'
        #print n_input
        
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                1,
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        '''
        print "dimension"
        print str(filter_sizes[layer_i]) + " " + str(n_input) + " " + str(n_output)
        '''
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output =   avg_pool_1x4(
                    lrelu(
                    tf.add(
                    tf.nn.conv2d(current_input, W, 
                                strides=[1,1,1,1], padding='SAME'), b)))
        #print output.get_shape().as_list()
        current_input = output
    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    a = [1]*784
    train_np_image = [a,a,a,a,a,a,a,a,a,a,a]
    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        #print un_avg_pool_1x4(current_input).get_shape().as_list()
        '''
        batch_0 = tf.shape(current_input)[0]
        batch_1 = tf.shape(un_avg_pool_1x4(current_input))[0]
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        batch_size_0 = sess.run(batch_0, {x: train_np_image})
        batch_size_1 = sess.run(batch_1, {x: train_np_image})
        print batch_size_0
        print batch_size_1
        '''

        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                #current_input, W,
                un_avg_pool_1x4(current_input), W,
                #W, current_input,
                tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1,1,1,1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))
    #print cost
    encoder.reverse

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'encoder': encoder}

'''
ae = build_ae()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
list = sess.run(ae['encoder'])
'''