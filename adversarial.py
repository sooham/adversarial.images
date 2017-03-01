# Generates adversarial image examples

# function to get correctly generated examples in test dataset

# function to backpropagate on image vectors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as path
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import covnet_model
from covnet_model import savedir

########################### HYPER PARAMETERS ###################################
# hyper parameter trials
# 0.03 23.1 17.3 37.2 32.8 31.0 16.3 28.6
# 0.04 29.1 28.2 28.4 19.6 30.4 20.2 15.2
# 0.05 20.4 32.5 28.1 40.4 33.1 28.5 46.9
# 0.06 19.2 29.7 27.0 27.1 19.3 36.2 38.8
# 0.08 19.1 37.8 33.2 21.8 25.5 19.5 28.2
# 0.1  29.1 35.9 37.8 33.4 36.2 40.7 30.2
# 1.   46.2 58.2 50.7 64.9 46.3 47.8 63.1
learning_rate = 0.04
source = 2 # the index of class we want to take examples from
target = 6 # the index of class we want to generate adversarial examples of

###################### BUILD ADVERSARIAL GRAPH #################################


scaled_logits = tf.nn.softmax(covnet_model.pred)

src_acc = scaled_logits[0, source]

trgt_acc = scaled_logits[0, target]

# run gradient descent on element of scaled_logits vector corresponding
# to target adversarial class.
grad = tf.reshape(tf.gradients(trgt_acc, covnet_model.x)[0], shape=[1, 28, 28, 1])

init = tf.initialize_all_variables()

###################### SUMMARIES ###############################################

tf.image_summary('input', tf.reshape(covnet_model.x, shape=[1, 28, 28, 1]))
tf.image_summary('grad', grad)
tf.histogram_summary('grad', grad)
tf.scalar_summary('target_accuracy', trgt_acc)
tf.scalar_summary('source_accuracy', src_acc)

###################### TRAINING SCHEMA #########################################

if __name__ == '__main__':
  if not path.isdir(savedir):
    # train the model first
    covnet_model.train()

  fname = raw_input('Enter model saved filename > ')
  weights = path.join(savedir, fname)


  with tf.Session() as sess:
    sess.run(init)

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    covnet_model.saver.restore(sess, weights)
    print('Weights restored.')

    summaries = tf.merge_all_summaries()

    writer = tf.train.SummaryWriter(
        covnet_model.logdir,
        graph=sess.graph
    )


    # get all test data with class 1
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    twos = test_labels.argmax(axis=1) == source

    two_images = test_images[twos]
    two_labels = test_labels[twos]

    shuffle_idx = np.arange(two_images.shape[0])
    np.random.shuffle(shuffle_idx)

    two_images = two_images[shuffle_idx]
    two_labels = two_labels[shuffle_idx]

    print('Generating Adversarial Images.')
    print('Open tensorboard to visualize.')

    orig = two_images[np.newaxis, 0]
    image = np.copy(orig)
    label = two_labels[np.newaxis, 0]

    i = 0
    target_acc = 0.
    while target_acc < .99:
      logits, source_acc, target_acc, g, summ = sess.run(
        [scaled_logits, src_acc, trgt_acc, grad, summaries],
        feed_dict={covnet_model.x: image, covnet_model.y: label, covnet_model.keep_prob: 1.}
      )

      if i == 0 and not (np.argmax(logits) == 2):
        print('Randomly selected image not classified correctly. Try again.')
        sys.exit()

      writer.add_summary(summ, global_step=i)

      image = image + learning_rate * g.reshape(1, 28*28)

      diff = np.abs(orig - image)

      print("%d source_acc %.5f, target_acc %.5f, sum: %.5f" % (i, source_acc, target_acc, np.sum(diff)))

      i += 1

    print('Adversarial example generated.')

    fig = plt.figure(figsize=(12, 12))
    plt.subplot(131)
    plt.imshow(orig.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(diff.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

    if raw_input('Do you want to save this example? [y\N] ') == 'y':
      fname = raw_input('Enter name of .npy file to save to > ')
      np.savez(
        path.join(savedir, fname),
        source=orig,
        delta=diff,
        target=image,
        source_acc=source_acc,
        target_acc=target_acc
      )

  covnet_model.train_sess.close()
