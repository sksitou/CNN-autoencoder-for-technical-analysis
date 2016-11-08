import tensorflow as tf
import numpy as np
import math, sys, random
import matplotlib.pyplot as plt
from libs.activations import lrelu
from libs.utils import corrupt
import time_series_generator as ts
from conv_autoencoder import build_ae
'''
feed stocks with batch
'''

logs_path = '/tmp/run/1'
save_path = '/tmp/model.ckpt'
learning_rate = 0.01
batch_size = 2
n_epochs = 10

def run_nn():

    #train_np_image = [map(float,range(784)),map(float,range(784))]
    #train_np_image = [[float(i) for i in range(784)],[float(i) for i in range(784)]]
    #train_np_image = [[i*1.0 for i in range(784)],[i*1.0 for i in range(784)]*2]
    train_np_label = [range(10),range(10)]
    train_np_image = ts.create_list(4,784,ts.uptrend(ts.sinx))
    sample_size = len(train_np_image)
    print sample_size
    ae = build_ae()

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


    n_examples = 2
    '''
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    '''
    recon = sess.run(ae['y'], feed_dict={ae['x']: train_np_image})
    sample = [i[0][0] for i in recon[0].tolist()]
    index = range(len(sample))
    print len(sample)
    plt.plot(index, sample)
    #plt.plot(index, train_np_image[0])
    #plt.plot(index,sample,'bs',index,train_np_image[0],'r--')
    
    plt.show()
    #plt.waitforbuttonpress()
    sess.close()

if __name__ == '__main__':
    run_nn()