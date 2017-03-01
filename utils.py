# Utility functions

from __future__ import print_function

import os
import os.path as path

import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()

savedir = path.join(cwd, 'saves')
exampledir = path.join(cwd, 'examples')
logdir = path.join(cwd, 'logs')

def create_directories():
    '''
        Create directory rooted at cwd for every entry in dirs.
    '''

    for d in [savedir, logdir, exampledir]:
        if not path.exists(d): os.mkdir(d)

def get_class(cls, images, labels):
    ''' Returns array of all images and labels belonging to class cls
        which is randomly shuffled.
    '''

    cls_indexes = labels.argmax(axis=1) == cls
    cls_images = images[cls_indexes]
    cls_labels = labels[cls_indexes]

    shuffle_idx = np.arange(cls_labels.shape[0])
    np.random.shuffle(shuffle_idx)
    cls_images = cls_images[shuffle_idx]
    cls_labels = cls_labels[shuffle_idx]

    return cls_images, cls_labels

def generate_plot():
    # get all saved examples
    files = [path.join(exampledir, f) for f in os.listdir(exampledir) if f.endswith('.npz')]

    fig = plt.figure(figsize=(15, 48))

    for row in range(10):
        data = np.load(files[row])

        plt.subplot(10, 3, 3 * row + 1)
        plt.axis('off')
        plt.imshow(data['source'].reshape(28, 28), cmap='gray')

        plt.subplot(10, 3, 3 * row + 2)
        plt.axis('off')
        plt.imshow(data['delta'].reshape(28, 28), cmap='gray')

        plt.subplot(10, 3, 3 * row + 3)
        plt.axis('off')
        plt.imshow(data['target'].reshape(28, 28), cmap='gray')

    plt.show()