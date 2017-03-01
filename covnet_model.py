# Covnet for MINST data
# Uses tensorflow r0.11

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import create_directories

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import os.path as path
import sys
import tensorflow as tf

#################### SETUP ####################
savedir, logdir = create_directories(['save', 'logs'])

#################### HYPER-PARAMETERS ####################

learning_rate = 1e-4
n_iter = 10          # number of gradient descent iterations
batch_size = 50         # mini-batch size from dataset
dropout_prob = 0.5

#################### HELPER FUNCTIONS ####################


def convolution_layer(tensor, F, b, name, k=2):
    '''
        Adds a convolution layer with filter F and bias b
        with ReLU activations and maxpooling.

        Tensor needs to be of shape [batch, in_height, in_width, in_channels]
        and filter needs to be of shape
        [filter_height, filter_width, in_channels, out_channels].
    '''
    with tf.name_scope(name) as scope:
        tensor_conv = tf.nn.conv2d(
            tensor, F, strides=[1, 1, 1, 1], padding="SAME"
        )
        tensor_add_bias = tf.nn.bias_add(tensor_conv, b)
        activations = tf.nn.relu(tensor_add_bias)
        max_pool = tf.nn.max_pool(
            activations, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'
        )
        return max_pool


def weight(shape, name, stddev=0.1):
    '''
        Initializes weight of shape following truncated normal distribution.
    '''
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias(shape, name):
    ''' Initializes bias of given shape. '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

#################### COVNET GRAPH ####################


def covnet(tensor, weights, biases, keep_prob):
    '''
      Creates a 3 hidden layer covnet model for MNIST data.
      Return the model predictions on classes.

      tensor: MINST input vector [-1, 784].
      weights: dictionary of weights.
      biases: dictionary of biases.
      keep_prob: dropout probability Variable.
    '''

    # reshape input tensor to shape accepted by conv2d
    tensor_r = tf.reshape(tensor, shape=[-1, 28, 28, 1])
    tf.image_summary('Input Image', tensor_r)

    h1 = convolution_layer(tensor_r, weights['wc1'], biases['bc1'], 'Conv1')
    tf.histogram_summary('Convolution layer 1', h1)

    h2 = convolution_layer(h1, weights['wc2'], biases['bc2'], 'Conv2')
    tf.histogram_summary('Convolution layer 2', h2)

    # fully connected layer
    h2flat = tf.reshape(h2, [-1, 7*7*64])

    h3 = tf.nn.relu(tf.matmul(h2flat, weights['wf1']) + biases['bf1'])
    tf.histogram_summary('Fully connected layer', h3)

    h3drop = tf.nn.dropout(h3, keep_prob)

    # final output layer
    out = tf.matmul(h3drop, weights['wo1']) + biases['bo1']
    tf.histogram_summary('Model predictions', tf.nn.softmax(out))

    return out

#################### TF PLACEHOLDERS ####################

x = tf.placeholder(tf.float32, [None, 784], name='input_image_batch')
y = tf.placeholder(tf.float32, [None, 10], name='input_labels_batch')
keep_prob = tf.placeholder(tf.float32, name='dropout_probability')

#################### WEIGHTS AND BIASES ####################

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': weight([5, 5, 1, 32], 'filter_layer_1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': weight([5, 5, 32, 64], 'filter_layer_2'),
    # fully connected layer with 1024 hidden units and dropout
    'wf1': weight([7*7*64, 1024], 'fully_connected_weights'),
    # output layer
    'wo1': weight([1024, 10], 'out_weights')
}

biases = {
    'bc1': bias([32], 'filter_bias_1'),
    'bc2': bias([64], 'filter_bias_2'),
    'bf1': bias([1024], 'full_bias'),
    'bo1': bias([10], 'out_bias')
}

#################### LOSS COMPUTATION AND OPTIMIZATION ####################

pred = covnet(x, weights, biases, keep_prob)

with tf.name_scope('Loss_computation') as scope:

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred, y)
    )
    tf.scalar_summary('cross entropy', cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate
).minimize(cross_entropy)

with tf.name_scope('Accuracy_computation'):
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32)
  )
  tf.scalar_summary('accuracy', accuracy)

#################### SUMMARY LOGGING ####################

# create a Saver object to save all variables
saver = tf.train.Saver()
train_sess = tf.Session()

def log_summaries():
    ''' Logs training critical metrics. '''

    with tf.name_scope('Conv_Layer_1'):
        # display all filters in first layer
        filter_arr = tf.split(3, 32, weights['wc1'])
        filter_arr = [tf.reshape(f, shape=[1, 5, 5, 1]) for f in filter_arr]
        for i in range(len(filter_arr)):
            tf.image_summary('filter_%d' % i, filter_arr[i])

        tf.scalar_summary('wc1/mean', tf.reduce_mean(weights['wc1']))
        tf.scalar_summary('bc1/mean', tf.reduce_mean(biases['bc1']))

    with tf.name_scope('Conv_Layer_2'):
        # cannot show all images of filters due to size constraints
        tf.scalar_summary('wc2/mean', tf.reduce_mean(weights['wc2']))
        tf.scalar_summary('bc2/mean', tf.reduce_mean(biases['bc2']))

    with tf.name_scope('Full_Layer_1'):
        tf.scalar_summary('wf1/mean', tf.reduce_mean(weights['wf1']))
        tf.scalar_summary('bf1/mean', tf.reduce_mean(biases['bf1']))

    with tf.name_scope('Out_Layer'):
        tf.scalar_summary('wo1/mean', tf.reduce_mean(weights['wo1']))
        tf.scalar_summary('bo1/mean', tf.reduce_mean(biases['bo1']))

#################### TRAINING SESSION ####################


def train():
    # start recording summaries
    log_summaries()
    summaries = tf.merge_all_summaries()

    writer = tf.train.SummaryWriter(
        logdir,
        graph=train_sess.graph
    )

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    init = tf.global_variables_initializer()

    train_sess.run(init)
    print('Initialized Variables...')

    print('Training...')
    print('Launch TensorBoard to see metrics.')

    for i in range(n_iter):
        batch = mnist.train.next_batch(batch_size)
        _, summ = train_sess.run(
          [optimizer, summaries],
          feed_dict={x: batch[0], y: batch[1], keep_prob: dropout_prob}
        )

        writer.add_summary(summ, global_step=i)

    print('')
    # done training, calculate acc on test set
    print("Test Accuracy: %g" % train_sess.run(
        accuracy,
        feed_dict={
            x: mnist.test.images,
            y: mnist.test.labels,
            keep_prob: 1.0
        }
    ))

    print('')

    # save if desired
    while True:
        prompt = raw_input('Do you wish to save model weights? [y/N] ')

        if prompt == 'y':
            fname = raw_input('Enter filename > ')
            save_path = saver.save(train_sess, path.join(savedir, fname))
            print('Model saved at ' + save_path)
            break
        elif prompt == 'N':
            break

    # close files and sessions
    writer.close()
    train_sess.close()


if __name__ == '__main__':
    train()
