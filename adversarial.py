# Generates adversarial images

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import covnet_model
from utils import create_directories, get_class, savedir, logdir, exampledir

#################### HYPER PARAMETERS ####################

learning_rate = 0.04

#################### BUILD ADVERSARIAL GRAPH ####################

def build_graph(source, target):
    with tf.name_scope('Adversarial') as scope:
        scaled_logits = tf.nn.softmax(covnet_model.pred)

        src_acc = scaled_logits[0, source]
        trgt_acc = scaled_logits[0, target]

        tf.scalar_summary('target_accuracy', trgt_acc)
        tf.scalar_summary('source_accuracy', src_acc)

        # run gradient descent on element of scaled_logits vector corresponding
        # to target adversarial class.
        grad = tf.reshape(tf.gradients(trgt_acc, covnet_model.x)[0], shape=[1, 28, 28, 1])
        tf.image_summary('grad', grad)
        tf.histogram_summary('grad', grad)

        return scaled_logits, src_acc, trgt_acc, grad

#################### TRAINING SCHEMA ####################

def train(source, target):

    scaled_logits, src_acc, trgt_acc, grad = build_graph(source, target)

    init = tf.global_variables_initializer()
    summaries = tf.merge_all_summaries()

    if not path.isdir(savedir):
        print('No models found. Start training.')
        covnet_model.train()

    create_directories()

    if raw_input('Do you want to use your own weights? [y\N] ') == 'y':
        fname = raw_input('Enter saved model name > ')
        weights = path.join(savedir, fname)
    else:
        weights = path.join(savedir, 'default')

    with tf.Session() as sess:
        sess.run(init)
        covnet_model.saver.restore(sess, weights)
        print('Weights restored.')

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        writer = tf.train.SummaryWriter(
            logdir,
            graph=sess.graph
        )

        src_images, src_labels = get_class(
            source,
            mnist.test.images,
            mnist.test.labels
        )

        # pick a random image that is correctly classified by CNN
        k = 0
        while True:
            original = src_images[np.newaxis, k]
            label = src_labels[np.newaxis, k]
            image = np.copy(original)

            l = scaled_logits.eval(feed_dict={
                    covnet_model.x: original,
                    covnet_model.y: label,
                    covnet_model.keep_prob: 1.
                    }
                )

            if np.argmax(l) == source:
                # correctly classified
                break

        print('Generating Adversarial Image...')
        print('Open tensorboard to visualize.')

        # train loop
        i = 0
        target_acc = 0.
        start_acc = []

        while target_acc < .99:  # fool to 99% acc
            source_acc, target_acc, dimg, summ = sess.run(
                [src_acc, trgt_acc, grad, summaries],
                feed_dict={
                    covnet_model.x: image,
                    covnet_model.y: label,
                    covnet_model.keep_prob: 1.
                }
            )

            if i == 0:
                start_acc.extend([source_acc, target_acc])

            writer.add_summary(summ, global_step=i)

            image = image + learning_rate * dimg.reshape(1, 28*28)

            diff = np.abs(original - image)

            print("%d  source_acc %.5f, target_acc %.5f, sum: %.5f" % (
                    i, source_acc, target_acc, np.sum(diff)
                )
            )

            i += 1

        print('Adversarial example generated.')

        # Show the example
        fig = plt.figure(figsize=(30, 10))

        plt.subplot(131)
        plt.imshow(original.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Original. source: (%f), target: (%f)' % tuple(start_acc))

        plt.subplot(132)
        plt.imshow(diff.reshape(28, 28), cmap='gray')
        plt.title('Delta (%f)' % np.sum(diff))
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Adversarial source: (%f), target: (%f)' % (source_acc, target_acc))

        plt.show()


        # ask to save
        while True:
            prompt = raw_input('Do you want to save this example? [y\N] ')

            if prompt == 'y':
                fname = raw_input('Enter name of npy file without extension > ')
                np.savez(
                    path.join(exampledir, fname),
                    source=original,
                    delta=diff,
                    target=image,
                    source_acc=source_acc,
                    target_acc=target_acc
                )
                break
            elif prompt == 'N':
                break

        covnet_model.train_sess.close()


if __name__ == '__main__':
    train(2, 6)
