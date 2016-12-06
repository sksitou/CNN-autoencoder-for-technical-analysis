import tensorflow as tf
import numpy as np
import math, sys, random
import matplotlib.pyplot as plt
#from plotly import tools
#import plotly.plotly as py
#import plotly.graph_objs as go

from libs.activations import lrelu
from libs.utils import corrupt,load_data,slice_data
import time_series_generator as ts
from conv_autoencoder import build_ae
'''
feed stocks with batch
'''

logs_path = '/tmp/run/1'
save_path = '/tmp/model_stock.ckpt'
file_name = '2833_out.csv'
learning_rate = 0.001
batch_size = 2
n_epochs = 1
input_size = 512
n_sample = 30
n_pooling = 1
corruption = ts.corruption
term = ts.uptrend(ts.sinx)
n_filters = 2

def run_nn():

    #ts.corruption=float(sys.argv[2])
    corruption=float((sys.argv[2]))
    save_path=str(n_filters) + '_' + str(corruption) + '.ckpt'  
    #train_np_image = [map(float,range(784)),map(float,range(784))]

    
    #a = [float(i) for i in range(input_size)]
    '''
    a = [1]*input_size
    feature1 = [a,a,a,a,a,a,a,a,a,a,a]
    

    #train_np_image = [[i*1.0 for i in range(784)],[i*1.0 for i in range(784)]*2]
    
    train_np_label = [range(10),range(10)]
    feature2 = ts.create_list(10,input_size,term,randomize=True)
    '''
    data = load_data(file_name,['Normalized_close'])[0]
    data_len = len(data)/5
    train_data, test_data = data[data_len*4:], data[:data_len]
    train_np_image = slice_data(train_data,input_size,n_sample)
    test_data = slice_data(test_data,input_size,20,constant=True)
    #print test_data

    #train_np_image = feature1 + feature2
    
    sample_size = len(train_np_image)

    #print sample_size
    ae = build_ae(input_size=input_size,
                input_shape=[None,input_size],
                n_pooling = n_pooling,
                corruption=False,
                corruption_level=corruption)
                #n_filters = n_filters)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    sess = tf.Session()
    saver = tf.train.Saver()
    if sys.argv[1] == 'train':
        sess.run(tf.initialize_all_variables())
    if sys.argv[1] == 'restore':
        saver.restore(sess, save_path)

    tf.scalar_summary('cost',ae['cost'])
    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())


    for epoch_i in range(n_epochs):
        #for batch_i in range(mnist.train.num_examples // batch_size):
        '''SGD'''
        for batch_i in range(1,sample_size // batch_size):
            #batch_xs, _ = mnist.train.next_batch(batch_size)
            random.shuffle(train_np_image)
            #print train_np_image[:batch_i]
            batch_xs = train_np_image[:batch_i]
            #train = np.array([img - mean_img for img in batch_xs])
            train = batch_xs
            _,summary = sess.run([optimizer,summary_op], feed_dict={ae['x']: train})
            #print ae['cost']
        writer.add_summary(summary,epoch_i * batch_i)
        #print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))
    saver.save(sess, save_path)
    
    output = sess.run(ae['y'], feed_dict={ae['x']: test_data})
    W = sess.run(ae['encoder'])
    if n_filters == 3:
        W1,W2,W3= W[0][0],W[1][0],W[2][0]
    else:
        W1,W2= W[0][0],W[1][0]
    x = range(len(output[0]))
    output1 = output[0].T.tolist()[0][0]
    layer1 = W1.T.tolist()[0][0]
    layer2 = W2.T.tolist()[0][0]

    if n_filters == 3:
        layer3 = W3.T.tolist()[0][0]
    
    scanner = range(len(layer1))

    
    #W1
    output0 = output[0].T.tolist()[0][0]
    #output0 = output[2].T.tolist()[0][0]

    for i, test in enumerate(test_data):
        plt.figure(100+i)
        output = output[i-1].T.tolist()[0][0]
        plt.plot(range(len(output)),output)
        plt.savefig('output2833'+str(i)+'.png')
        plt.figure(200+i)
        plt.plot(range(len(test)),test)
        plt.savefig('input2833'+str(i)+'.png')
    
    layer1 = W1.T.tolist()[0]
    index = len(layer1[0])
    for i, filters in enumerate(layer1):
        plt.figure(i)
        plt.plot(range(index),filters)
        plt.savefig('layer1,filter{i}.png'.format(i=i))
    layer2 = W2.T.tolist()[0]
    index = len(layer2[0])
    for i, filters in enumerate(layer1):
        plt.figure(i)
        plt.plot(range(index),filters)
        plt.savefig('layer2,filter{i}.png'.format(i=i))
    #plt.show()
    
    #plt.waitforbuttonpress()
    
    sess.close()
    return W,output[0], train_np_image[0]

if __name__ == '__main__':
    W,output,train = run_nn()