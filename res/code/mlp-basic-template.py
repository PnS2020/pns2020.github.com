"""Multi-Layer Perceptron for Fashion MNIST Classification.

Team #name
"""
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from pnslib import utils
from pnslib import ml

# Load all the ten classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels
(train_x, train_y, test_x, test_y) = utils.fashion_mnist_load(
    data_type="full", flatten=True)

num_classes = 10

print("[MESSAGE] Dataset is loaded.")

# preprocessing for training and testing images
train_x = train_x.astype("float32")/255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32")/255.
test_x -= mean_train_x

print("[MESSAGE] Dataset is preprocessed.")

# Use PCA to reduce the dimension of the dataset,
# so that the training will be less expensive
# perform PCA on training dataset
train_X, R, n_retained = ml.pca(train_x)

# perform PCA on testing dataset
test_X = ml.pca_fit(test_x, R, n_retained)

print("[MESSAGE] PCA is complete.")

# converting the input class labels to categorical labels for training
train_Y = to_categorical(train_y, num_classes=num_classes)
test_Y = to_categorical(test_y, num_classes=num_classes)

print("[MESSAGE] Converted labels to categorical labels.")

# sample sizes and feature dimensions
num_train_samples = train_X.shape[0]
num_test_samples = test_X.shape[0]
input_dim = train_X.shape[1]

# defining the batch size and epochs
batch_size = 64
num_epochs = 20

# defining the learning rate
lr = 0.1

# defining the number of layers and number of hidden units in each layer
num_layers = 2
num_units = [100, 100]

# build the model here
# provide the train_function which takes in the input and target, and outputs
# the loss, while updating the necessary variables underneath
# please provide the test_function which takes in the input and target , and
# outputs a tuple of (accuracy, prediction)
# >>>>> PUT YOUR CODE HERE <<<<<





# training loop here
num_batches = num_train_samples // batch_size
indices = list(range(num_train_samples))

for epoch in range(num_epochs):
    epoch_loss = 0.
    np.random.shuffle(indices)
    for batch_idx in range(num_batches):
        batch_indices = indices[batch_idx * batch_size:
                                (batch_idx + 1) * batch_size]
        batch_input = train_X[batch_indices]
        batch_target = train_Y[batch_indices]
        batch_loss, = train_function((batch_input, batch_target))
        epoch_loss += batch_loss
    epoch_loss /= num_batches
    print('Epoch {}, loss {} on {} batches'.format(
        epoch + 1, epoch_loss, num_batches))

num_batches = num_test_samples // batch_size
indices = range(num_test_samples)
overall_accuracy = 0.
for batch_idx in range(num_batches):
    batch_indices = indices[batch_idx * batch_size:
                            (batch_idx + 1) * batch_size]
    batch_input = test_X[batch_indices]
    batch_target = test_Y[batch_indices]
    batch_accuracy, batch_prediction = test_function(
        (batch_input, batch_target))
    overall_accuracy += batch_accuracy.mean()
overall_accuracy /= num_batches
print('Overall test accuracy {}'.format(overall_accuracy))

# visualize the ground truth and prediction
# take first 64 examples in the testing dataset
# note that we are processing the 64 samples here, instead of just the 10
# examples like in the keras example
test_X_vis = test_X[:batch_size]  # fetch first 64 samples
# fetch first 64 ground truth prediction
ground_truths = test_y[:batch_size]
ground_truth_categorical = test_Y[:batch_size]
# predict with the model
preds = test_function((test_X_vis,
                       ground_truth_categorical))[1]
preds = preds.astype(np.int)

labels = ["Tshirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle Boot"]

plt.figure()
for i in range(2):
    for j in range(5):
        plt.subplot(2, 5, i*5+j+1)
        plt.imshow(test_x[i*5+j].reshape(28, 28), cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[ground_truths[i*5+j]],
                   labels[preds[i*5+j]]))
plt.show()
