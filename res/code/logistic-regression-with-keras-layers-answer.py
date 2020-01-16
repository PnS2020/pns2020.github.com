"""Logistic Regression for Fashion MNIST Binary Classification.

Team #name
"""
from __future__ import print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from pnslib import utils
from pnslib import ml

# Load T-shirt/top and Trouser classes from Fashion MNIST
# complete label description is at
# https://github.com/zalandoresearch/fashion-mnist#labels
(train_x, train_y, test_x, test_y) = utils.binary_fashion_mnist_load(
    class_list=[0, 1],
    flatten=True)

print("[MESSAGE] Dataset is loaded.")


# preprocessing for training and testing images
train_x = train_x.astype("float32")/255.  # rescale image
mean_train_x = np.mean(train_x, axis=0)  # compute the mean across pixels
train_x -= mean_train_x  # remove the mean pixel value from image
test_x = test_x.astype("float32")/255.
test_x -= mean_train_x

print("[MESSAGE] Dataset is preporcessed.")

# Use PCA to reduce the dimension of the dataset,
# so that the training will be less expensive
# perform PCA on training dataset
train_X, R, n_retained = ml.pca(train_x)

# perform PCA on testing dataset
test_X = ml.pca_fit(test_x, R, n_retained)

print("[MESSAGE] PCA is complete")

# define a model
num_train_samples = train_X.shape[0]
num_test_samples = test_X.shape[0]
input_dim = train_X.shape[1]

x = Input(shape=(input_dim,))
y = Dense(1, activation="sigmoid")(x)
model = Model(x, y)

print("[MESSAGE] Model is defined.")

# print model summary
model.summary()

# compile the model aganist the binary cross entropy loss
# and use SGD optimizer
model.compile(loss="binary_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

print("[MESSAGE] Model is compiled")

# train the model
history = model.fit(
    x=train_X, y=train_y,
    batch_size=64, epochs=10,
    validation_data=(test_X, test_y))

# save the trained model
model.save("logistic-regression-fashion-mnist-trained.hdf5")

# visualize the ground truth and prediction
# take first 10 examples in the testing dataset
test_X_vis = test_X[:10]  # fetch first 10 samples
ground_truths = test_y[:10]  # fetch first 10 ground truth prediction
# predict with the model
preds = (model.predict(test_X_vis) > 0.5)[:, 0].astype(np.int)

labels = ["Tshirt/top", "Trouser"]

plt.figure()
for i in range(2):
    for j in range(5):
        plt.subplot(2, 5, i*5+j+1)
        plt.imshow(test_x[i*5+j].reshape(28, 28), cmap="gray")
        plt.title("Ground Truth: %s, \n Prediction %s" %
                  (labels[ground_truths[i*5+j]],
                   labels[preds[i*5+j]]))
plt.show()
