#!/usr/bin/env python

###########################################################################
# learn.py: Machine learning algorithms for TensorFlow skills test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-28
###########################################################################

import data_preprocessing
import graph_construction

import keras
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

class BatchHistory(keras.callbacks.Callback):
    """Keras callback for recording validation accuracy at regular
    intervals, i.e. every set number of batches.  This should be
    passed to a 'fit' call via the list passed to its 'callbacks'
    keyword argument, and the model must be compiled to have
    'accuracy' as one of its metrics.

    After this, the 'history' property will contain a list of
    dictionaries.  Each dictionary will have the following keys and
    values in it:
    batch -- batch number for all loss & accuracy numbers
    val_loss -- validation loss
    val_acc -- validation accuracy
    loss -- training loss
    acc -- training accuracy
    """
    def __init__(self, test_X, test_y, skip=100):
        """Initialize a BatchHistory.

        Parameters:
        skip -- Interval (of batch number) at which to record accuracy
        test_X -- Input data for predicting accuracy
        test_y -- Correct labels corresponding to test_X
        """
        self.skip = skip
        self.history = []
        # Below are required for evaluating:
        self.test_X = test_X
        self.test_y = test_y
        # 'i' tallies up the batch number:
        self.i = 0
        # These two lists store the past training accuracies and
        # losses for up to 'self.skip' batches, and they are
        # periodically used to provide an averaged training loss &
        # accuracy over several batches, and then cleared.
        self.losses = []
        self.accs = []
    def on_batch_end(self, batch, logs={}):
        self.i += 1
        self.accs.append(logs["acc"])
        self.losses.append(logs["loss"])
        if self.i % self.skip == 0:
            ev = self.model.evaluate(self.test_X, self.test_y, verbose=0)
            l = {}
            l["loss"] = sum(self.losses) / len(self.losses)
            l["acc"] = sum(self.accs) / len(self.accs)
            self.losses = []
            self.accs = []
            l["batch"] = self.i
            l["val_loss"] = ev[0]
            l["val_acc"] = ev[1]
            self.history.append(l)

def train_model(num_epochs, batch_size, learning_rate):
    """Trains a neural network for image classification from the SVHN
    dataset, and creates a plot giving training/testing accuracy as a
    function of batch number.

    Parameters:
    num_epochs -- Number of training epochs
    batch_size -- Number of examples in each training batch
    learning_rate -- Initial learning rate
    """
    # Get data:
    train_X_orig, train_y_orig, _, _ = data_preprocessing.load_data()
    train_X_norm = data_preprocessing.normalize(train_X_orig)
    train_X, valid_X, train_y, valid_y = data_preprocessing.split(
        train_X_norm, train_y_orig)
    # One-hot encode so they can be used for input/validation:
    train_y_cat = keras.utils.to_categorical(train_y, num_classes=10)
    valid_y_cat = keras.utils.to_categorical(valid_y, num_classes=10)
    
    # Build & compile model:
    model = graph_construction.get_keras_model()
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    # Add an extra dimension to fit with model's input format:
    train_X2 = np.expand_dims(train_X, axis=3)
    valid_X2 = np.expand_dims(valid_X, axis=3)

    # Train:
    history = BatchHistory(valid_X2, valid_y_cat)    
    model.fit(train_X2,
              train_y_cat,
              epochs=num_epochs,
              batch_size=batch_size,
              callbacks=[acc_history],
              validation_data=(valid_X2, valid_y_cat))
    # The callback slows things down a bit, but I'm not sure of a good
    # way around it.  If I were testing only on specific batches of
    # validation data, it might be less of an issue.

    fname_base = "model_{0}_{1}_{2}".format(learning_rate, num_epochs,
                                            batch_size)
    model.save_weights("{0}.h5".format(fname_base))

    # Plot training/validation curve:
    b = [i[0] for i in history.history]
    plt.plot(b, [i[2] for i in history.history])
    plt.plot(b, [i[1] for i in history.history])
    plt.ylabel('Accuracy')
    plt.xlabel('Batch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig("{0}.png".format(fname_base))
    plt.show()

if __name__ == '__main__':
    train_model(5, 64, 0.02)
