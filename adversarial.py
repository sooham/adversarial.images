# Generates adversarial images for a given class in a given dataset.
# Use tensorflow version 0.11

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as path
import sys
import numpy as np
import tensorflow as tf

# Import data
from tensorflow.examples.tutorials.mnist import input_data

cwd = os.getcwd()
logdir = path.join(cwd, 'logs')
savedir = path.join(cwd, 'save')

if not path.exists(logdir):
  os.mkdir(logdir)
if not path.exists(savedir):
  os.mkdir(savedir)

# TODO: set a seed for all random operations

############### HYPER-PARAMETERS ###########################
learning_rate = 1e-4
n_iter = 20000     # number of gradient descent iterations
batch_size = 50    # mini-batch size from dataset
dropout_prob = 0.5

############### HELPER FUNCTIONS ###########################
def conv2d(tensor, F, b, strides=1):
  '''
    Adds a convolutional layer with filter F and bias b
    and ReLU activations. tensor needs to be of shape
    [batch, in_height, in_width, in_channels] and filter
    needs to be of shape
    [filter_height, filter_width, in_channels, out_channels].
  '''
  tensor_conv = tf.nn.conv2d(tensor, F, strides=[1, strides, strides, 1], padding="SAME")
  tensor_add_bias = tf.nn.bias_add(tensor_conv, b)
  return tf.nn.relu(tensor_add_bias)

def maxpool(tensor, k=2):
  ''' Wrapper to compute maxpool with kxk receptive field. '''
  return tf.nn.max_pool(
    tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'
  )

def weight(shape):
    ''' Initializes weight of given shape following truncated normal distribution. '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    ''' Initializes bias of given shape. '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

################# COVNET MODEL #############################

def covnet(tensor, weights, biases, keep_prob):
  '''
    Creates a 3 layer covnet model for the purposes of this
    assignment. Returns the model predictions on MINST classes.

    tensor - MINST input vector [-1, 784]
    weights - dictionary of weights
    biases - dictionary of biases
    keep_prob - tf.float32 placeholder representing dropout probability
  '''

  # reshape input tensor to shape accepted by conv2d
  tensor_r = tf.reshape(tensor, shape=[-1, 28, 28, 1])

  # add summaries to analyze in tensorboard
  tf.image_summary('MINST input image', tensor_r)

  # first convolution layer
  h_layer_1_conv = conv2d(tensor_r, weights['wc1'], biases['bc1'])

  tf.histogram_summary('Convolution layer 1 conv', h_layer_1_conv)

  h_layer_1_pool = maxpool(h_layer_1_conv)
  tf.histogram_summary('Convolution layer 1 pool', h_layer_1_pool)

  # second convolution layer
  h_layer_2_conv = conv2d(h_layer_1_pool, weights['wc2'], biases['bc2'])

  tf.histogram_summary('Convolution layer 2 conv', h_layer_2_conv)

  h_layer_2_pool = maxpool(h_layer_2_conv)
  tf.histogram_summary('Convolution layer 2 pool', h_layer_2_pool)

  # fully connected layer
  h_layer_2_pool_flat = tf.reshape(h_layer_2_pool, [-1, 7*7*64])

  h_fully_connected = tf.nn.relu(tf.matmul(h_layer_2_pool_flat, weights['wf1']) + biases['bf1'])
  tf.histogram_summary('Fully connected RELU', h_fully_connected)

  h_fc1_drop = tf.nn.dropout(h_fully_connected, keep_prob)

  # final output layer
  out = tf.matmul(h_fc1_drop, weights['wo1']) + biases['bo1']
  tf.histogram_summary('Model predictions', out)

  return out

################## TF Graph inputs #########################

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

################### weights and biases #####################

weights = {
  # 5x5 conv, 1 input, 32 outputs
  'wc1': weight([5, 5, 1, 32]),
  # 5x5 conv, 32 inputs, 64 outputs
  'wc2': weight([5, 5, 32, 64]),
  # fully connected layer with 1024 hidden units and dropout
  'wf1': weight([7*7*64, 1024]),
  # output layer
  'wo1': weight([1024, 10])
}

biases = {
  'bc1': bias([32]),
  'bc2': bias([64]),
  'bf1': bias([1024]),
  'bo1': bias([10])
}


############## Loss Computation and Optimizer ##############

pred = covnet(x, weights, biases, keep_prob)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.GradientDescentOptimizer(
  learning_rate=learning_rate
).minimize(cross_entropy)

accuracy = tf.reduce_mean(
  tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32)
)

################# Summary logging #########################

# create a Saver object to save all variables for later use
saver = tf.train.Saver()

with tf.name_scope('Layer_1'):
  filter_arr = tf.split(3, 32, weights['wc1'])
  filter_arr = [tf.reshape(f, shape=[1, 5, 5, 1]) for f in filter_arr]
  for i in range(len(filter_arr)):
    tf.image_summary('filter_%d' % i, filter_arr[i])

  tf.scalar_summary('bc1/mean', tf.reduce_mean(biases['bc1']))

with tf.name_scope('Layer_2'):
  # cannot show all images of filters due to size constraints
  tf.scalar_summary('wc2/mean', tf.reduce_mean(weights['wc2']))
  tf.scalar_summary('bc2/mean', tf.reduce_mean(biases['bc2']))

with tf.name_scope('Full_Layer'):
  tf.scalar_summary('wf1/mean', tf.reduce_mean(weights['wf1']))
  tf.scalar_summary('bf1/mean', tf.reduce_mean(biases['bf1']))

with tf.name_scope('Out_Layer'):
  tf.scalar_summary('wo1/mean', tf.reduce_mean(weights['wo1']))
  tf.scalar_summary('bo1/mean', tf.reduce_mean(biases['bo1']))

tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('cross entropy', cross_entropy)

summaries = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(
    logdir,
    graph=optimizer.graph
)

################# Training session #########################

init = tf.initialize_all_variables()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session() as sess:
  sess.run(init)

  print('Training covnet model on MINST...')
  print('Open tensorboard to visualize.')

  for i in range(n_iter):
    batch = mnist.train.next_batch(batch_size)
    _, summ = sess.run(
      [optimizer, summaries],
      feed_dict={x: batch[0], y: batch[1], keep_prob: dropout_prob}
    )

    writer.add_summary(summ, global_step=i)

  # done training, calculate metrics on test set
  print("test accuracy %g" % accuracy.eval(
    feed_dict={
      x: mnist.test.images,
      y: mnist.test.labels,
      keep_prob: 1.0
    }
  ))

  print("Optimization finished!")
  writer.close()

  # save if desired
  if raw_input('Do you want to save model weights? [y/N]') == 'y':
    fname = raw_input('Enter filename\n')
    save_path = saver.save(sess, path.join(savedir, fname))
    print('Model saved at: ' + save_path)