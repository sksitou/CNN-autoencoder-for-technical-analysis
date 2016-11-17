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
input_shape=[None, 784]
n_filters=[1, 1, 1]
filter_sizes=[5, 5, 5, 5]
corruption=False

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

x = tf.placeholder(
    tf.float32, input_shape, name='x')

if len(x.get_shape()) == 2:
    x_dim = np.sqrt(x.get_shape().as_list()[1])
    if x_dim != int(x_dim):
        raise ValueError('Unsupported input dimensions')
    x_dim = int(x_dim)
    x_tensor = tf.reshape(
        #x, [-1, x_dim, x_dim, n_filters[0]])
        x, [-1, 784, 1, n_filters[0]])
elif len(x.get_shape()) == 5:
    x_tensor = x
else:
    raise ValueError('Unsupported input dimensions')

current_input = x

if corruption:
    current_input = corrupt(current_input)

# %%
# Build the encoder
def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1,1,4,1], strides=[1,1,4,1], padding='SAME')
def avg_pool_1x4(x):
    return tf.nn.avg_pool(x, ksize=[1,4,1,1], strides=[1,4,1,1], padding='VALID')
def un_avg_pool_1x4(x):
    shape = tf.shape(x)[0]
    return tf.tile(x,tf.pack([shape,4,1,1]))


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
    output = avg_pool_1x4(
                lrelu(
                tf.add(
                tf.nn.conv2d(current_input, W, 
                            strides=[1,1,1,1], padding='SAME'), b)))
    print tf.shape(output).eval()
    current_input = output
# %%
# store the latent representation
z = current_input
encoder.reverse()
shapes.reverse()

# %%
# Build the decoder using the same weights

for layer_i, shape in enumerate(shapes):
    W = encoder[layer_i]
    b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
    print tf.shape(current_input)
    output = lrelu(tf.add(
        tf.nn.conv2d_transpose(
            un_avg_pool_1x4(current_input), W,
            #W, current_input,
            tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
            strides=[1,1,1,1], padding='SAME'), b))
    current_input = output


# 
# now have the reconstruction through the network

y = current_input
# cost function measures pixel-wise difference
cost = tf.reduce_sum(tf.square(y - x_tensor))
#print cost
encoder.reverse

# 


