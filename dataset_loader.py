'''
    Loading Datasets And do the data Preprocessing Work
'''

import keras
from collections import namedtuple

def load_data(name, norm = True):
    """
    Load the data
    name - the name of the dataset
    return object with data and labels
    """
    print ('Loading Data...')
    if name.lower()=='mnist':
        nb_classes = 10
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    elif name.lower()=='cifar10':
        nb_classes = 10
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    elif name.lower() == 'cifar100':
        nb_classes = 10
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data()
    elif name.lower() == 'fashion_mnist':
        nb_classes = 10
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    else:
        ValueError("Do not support the name of dataset.")

    Y_train = keras.utils.np_utils.to_categorical(train_labels, nb_classes).astype('float32')
    Y_test = keras.utils.np_utils.to_categorical(test_labels, nb_classes).astype('float32')
    if norm:
        train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.
        test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.

    Dataset = namedtuple('Dataset', ['X', 'Y', 'y', 'nb_classes'])
    trn = Dataset(train_images, Y_train, train_labels, nb_classes)
    tst = Dataset(test_images, Y_test, test_labels, nb_classes)
    del train_images, test_images, Y_train, Y_test, train_labels, test_labels

    return trn, tst