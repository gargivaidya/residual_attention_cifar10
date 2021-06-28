import os
import pickle
import numpy as np
import torch

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    training_files = [
        os.path.join(data_dir, 'data_batch_%d' % i)
        for i in range(1, 6)
    ]

        

    # Load the training dataset
    x_train = []
    y_train = []
    for training_file in training_files:    
        with open(training_file, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        x_train.append(d[b'data'].astype(np.float32))
        y_train.append(np.array(d[b'labels'], dtype=np.int32))
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    testing_file = os.path.join(data_dir, 'test_batch')
    with open(testing_file, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    x_test = d[b'data'].astype(np.float32)
    y_test = np.array(d[b'labels'], dtype=np.int32)

    
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE

    # Load the testing dataset\
    testing_file = os.path.join(data_dir, 'private_test_images.npy')

    x_test = torch.from_numpy(np.load(testing_file))
    x_test = x_test.permute(0,3,1,2)   

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE

    split_index = int(train_ratio*50000)

    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

