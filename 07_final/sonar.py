# A SONAR loader.

import numpy as np
import gzip
import struct
import sys

def load_patterns(filename):
    return np.genfromtxt(filename, delimiter=",", usecols=range(60))
    # Open and unzip the file of images:
    # with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables:
        # _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        # all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        # return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    # Insert a column of 1s in the position 0 of X.
    # (“axis=1” stands for: “insert a column, not a row”)
    return np.insert(X, 0, 1, axis=1)

# 48 patterns, each 61 elements (1 bias + 60 frequencies)
X_train = prepend_bias(load_patterns("./sonar.train"))

print("Train: %s" % X_train)

# 10000 images, each 785 elements, with the same structure as X_train
X_test = prepend_bias(load_patterns("./sonar.test"))

print("Test: %s" % X_test)

def load_labels(filename):
    return np.genfromtxt(filename, delimiter=",", usecols=(60), dtype=int).reshape(-1, 1)

    # # Open and unzip the file of images:
    # with gzip.open(filename, 'rb') as f:
    #     # Skip the header bytes:
    #     f.read(8)
    #     # Read all the labels into a list:
    #     all_labels = f.read()
    #     # Reshape the list of labels into a one-column matrix:
    #     return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def one_hot_encode(Y):
    print("Y: %s" % Y)
    n_labels = Y.shape[0]
    print("n_labels: %s" % n_labels)
    n_classes = 2
    encoded_Y = np.zeros((n_labels, n_classes))
    print("encoded_Y: %s" % encoded_Y)
    for i in range(n_labels):
        label = Y[i]
        print("i: %s, label: %s" % (i, label))
        encoded_Y[i][label] = 1
    return encoded_Y

# 60K labels, each a single digit from 0 to 9
Y_train_unencoded = load_labels("./sonar.train")

# 60K labels, each consisting of 10 one-hot encoded elements
Y_train = one_hot_encode(Y_train_unencoded)

print("Y_train" % Y_train)
sys.exit()

# 10000 labels, each a single digit from 0 to 9
Y_test = load_labels("../data/mnist/t10k-labels-idx1-ubyte.gz")
