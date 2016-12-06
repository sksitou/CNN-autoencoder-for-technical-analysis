import tensorflow as tf
import numpy as np
import math, sys, random, itertools
import matplotlib.pyplot as plt
#from plotly import tools
#import plotly.plotly as py
#import plotly.graph_objs as go

from libs.activations import lrelu
from libs.utils import corrupt,load_data,slice_data
import time_series_generator as ts
from conv_2c_autoencoder import build_ae
'''
feed stocks with batch
'''

logs_path = '/tmp/run/1'
save_path = '/tmp/model_stock.ckpt'
file_name = '2833_out.csv'
learning_rate = 0.001
batch_size = 2
n_epochs = 100
input_size = 512
n_sample = 30
n_pooling = 1
corruption = ts.corruption
term = ts.uptrend(ts.sinx)
n_filters = 3

def run_nn():

    #ts.corruption=float(sys.argv[2])
    corruption=float((sys.argv[2]))
    save_path='2c' + str(n_filters) + '_' + str(corruption) + 'random.ckpt'
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
    test_data = slice_data(train_data,input_size,20,constant=True)
    #print test_data

    #train_np_image = feature1 + feature2
    train_np_image = [[[1,10]]*512 for i in range(10)]  
    sample_size = len(train_np_image)

    #print sample_size
    ae = build_ae(input_size=input_size,
                input_shape=[None,input_size,2],
                n_pooling = n_pooling,
                corruption=True,
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
        '''
        if epoch_i%10 == 0:
            print ae['cost']
        '''
        writer.add_summary(summary,epoch_i * batch_i)
        #print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))
    saver.save(sess, save_path)

    '''
    output = sess.run(ae['y'], feed_dict={ae['x']: train_np_image})
    sample = [i[0][0] for i in output[0].tolist()]
    index = range(len(sample))
    print len(sample)
    plt.plot(index, sample)
    #plt.plot(index, train_np_image[0])
    #plt.plot(index,sample,'bs',index,train_np_image[0],'r--')
    '''

    
    output = sess.run(ae['y'], feed_dict={ae['x']: train_np_image})
    W = sess.run(ae['encoder'])
    if n_filters == 3:
        W1,W2,W3= W[0],W[1],W[2]
    else:
        W1,W2= W[0],W[1]
    x = range(len(output[0]))
    output1 = output[0].T.tolist()[0][0]
    layer1 = W1.T.tolist()
    layer2 = W2.T.tolist()

    if n_filters == 3:
        layer3 = W3.T.tolist()

    scanner = range(len(layer1))

    
    #trace1 = go.Scatter(x = x, y = output[0])
    #trace00 = go.Scatter(x = x, y = train_np_image[0])
    '''
    size = len(data)/input_size
    length = size*input_size
    data_plot = data[:length]
    trace00 = go.Scatter(x = range(len(data)), y = data_plot)
    data_output = slice_data(data_plot,512,size)
    output_stock = []
    for i in data_output:        
        out = sess.run(ae['y'], feed_dict={ae['x']: data})
        output_stock.append(out)
    '''
    '''
    trace00 = go.Scatter(x = range(len(data)), y = train_np_image[0])
    trace0 = go.Scatter(x = range(len(output1)), y = output1)
    trace1 = go.Scatter(x = scanner, y = layer1)
    trace2 = go.Scatter(x = scanner, y = layer2)
    if n_filters == 3:
        trace3 = go.Scatter(x = scanner, y = layer3)
    else:
        trace3 = go.Scatter(x = scanner, y = layer2)
    '''
    '''
    fig = tools.make_subplots(rows=5, cols=1, 
                            subplot_titles=('Input','Output','Hidden Layer1',
                                            'Hidden Layer2', 'Hidden Layer3'))
    '''
    '''

    fig = tools.make_subplots(rows=4, cols=1, 
                            subplot_titles=('Input','Output','Hidden Layer1',
                                            'Hidden Layer2'))
    fig.append_trace(trace00, 1, 1)
    fig.append_trace(trace0, 2, 1)
    fig.append_trace(trace1, 3, 1)
    fig.append_trace(trace2, 4, 1)
    #fig.append_trace(trace3, 5, 1)
    #print n_filters
    title = 'filters: {n_filters}, c:{corruption}'.format(n_filters=n_filters,corruption=ts.corruption)
    fig['layout'].update(height=600, width=600, title=title)
    plot_url = py.plot(fig, filename='stock')
    '''
    
    #W1
    output0 = output[0].T.tolist()
    #output0 = output[2].T.tolist()[0][0]
    
    
    plt.figure(1)
    plt.plot(range(len(output0[0][0])), output0[0][0])
    print output0[0][0]
    #plt.savefig('output.png')
    plt.figure(2)
    #plt.plot(range(len(train_np_image[20][0])),train_np_image[20][0])
    plt.plot(range(len(output0[1][0])),output0[1][0])
    print output0[1][0]
    #plt.savefig('input.png')
    
    layer1 = W1.T.tolist()
    index = len(layer1[0][0])
    plt.figure(10)
    f, axarr1 = plt.subplots(len(layer1))
    '''
    for i, filters in enumerate(layer1):
        filters = list(itertools.chain.from_iterable(filters))
        #print filters
        #plt.figure(100+i)
        #plt.plot(range(index),filters)
        axarr1[i-1].plot(range(index),filters)
        axarr1[i-1].set_xticklabels([])
        axarr1[i-1].set_yticklabels([])
        #plt.savefig('layer1,filter{i}.png'.format(i=i))
    #plt.savefig('layer1random.png')
    layer2 = W2.T.tolist()
    index = len(layer2[0][0])
    plt.figure(20)
    f, axarr2 = plt.subplots(len(layer2),len(layer2[0]))
    for i, filters in enumerate(layer2):
        for j, filters_2 in enumerate(filters):
            #filters_2 = list(itertools.chain.from_iterable(filters))
            #print filters_2
            #plt.figure(200+i*10+j)
            #plt.plot(range(index),filters_2)
            axarr2[i-1][j-1].plot(range(index),filters_2)
            axarr2[i-1][j-1].set_xticklabels([])
            axarr2[i-1][j-1].set_yticklabels([])
            #plt.savefig('layer2,filter{i}:{j}.png'.format(i=i,j=j))
    #plt.savefig('layer2random.png')
    #plt.savefig('layer2.png')
    #print layer3
    layer3 = W3.T.tolist()
    index = len(layer3[0][0])
    plt.figure(30)
    f, axarr3 = plt.subplots(len(layer2),len(layer2[0]))
    for i, filters in enumerate(layer3):
        for j, filters_2 in enumerate(filters):
            #plt.figure(300+i*10+j)
            #plt.plot(range(index),filters_2)
            axarr3[i-1][j-1].plot(range(index),filters_2)
            axarr3[i-1][j-1].set_xticklabels([])
            axarr3[i-1][j-1].set_yticklabels([])
            #plt.savefig('layer3,filter{i}:{j}.png'.format(i=i,j=j))
    #plt.savefig('layer3random.png')
    '''
    plt.show()
    
    #plt.waitforbuttonpress()
    
    sess.close()
    return W,layer1,layer2,layer3,output, train_np_image[0]

if __name__ == '__main__':
    W,layer1,layer2,layer3,output,train = run_nn()