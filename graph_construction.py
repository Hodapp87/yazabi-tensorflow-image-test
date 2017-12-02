#!/usr/bin/env python

###########################################################################
# graph_constructor.py: Computational graphs for image recognition skills test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-28
###########################################################################

import tensorflow as tf

import keras
import keras.backend as K
from keras.layers import Dropout, Flatten, Input, merge, Dense, Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.objectives import categorical_crossentropy
from keras.optimizers import SGD

def lrn(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
    """Returns a Keras layer implementing Local Response Normalization as
    described in the AlexNet paper.
    """
    # Adapted from:
    # https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/customlayers.py#L9
    def f(X):
        b, r, c, ch = X.shape
        half = n // 2
        square = keras.backend.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)), ((0, 0), (half, half)))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, :, :, i:i + int(ch)]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

def get_keras_model(input_shape=(32, 32, 1), output_count=10, use_dropout=True):
    """Returns a Keras model for the image classification on this dataset.
    This is based heavily on AlexNet, but removes one of the initial
    deep layers due to the smaller image sizes here.

    Parameters:
    input_shape -- Tuple with the model's input dimensions - default (32,32,1)
    output_count -- Integer for number of output classes - default 10
    use_dropout -- Boolean for whether to use dropout in training (default True)
    """
    inputs = Input(shape=input_shape)

    # Convolutional layers:
    
    #conv_1 = Conv2D(96, kernel_size=(11,11), strides=(4, 4), activation='relu', name='conv_1')(inputs)
    #conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    #conv_2 = lrn(name='convpool_1')(conv_2)
    #conv_2 = ZeroPadding2D((2, 2))(conv_2)
    #conv_2 = Conv2D(256, kernel_size=(5,5), activation='relu', name='conv_2')(conv_2)
    
    # Names are retained to be clear how original AlexNet translates:
    conv_2 = Conv2D(256, kernel_size=(5,5), activation='relu', name='conv_2')(inputs)
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = lrn()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv_3')(conv_3)
    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv_4')(conv_4)
    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv_5')(conv_5)

    # Fully-connected layers:
    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(256, activation='relu', name='dense_1')(dense_1)
    if use_dropout:
        dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(256, activation='relu', name='dense_2')(dense_2)
    if use_dropout:
        dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(output_count, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)
    
    model = Model(inputs=[inputs], outputs=[prediction])
    return model

def classifier(learning_rate, use_dropout):
    """Builds (but does not execute) a TensorFlow graph for image
    classification on this dataset.
    
    Arguments:
    learning_rate - Learning rate to use
    use_dropout - Boolean for whether this network should use dropout

    Returns:
    model -- Output of tf.global_variables_initializer()
    train_op -- TensorFlow node for optimizing this network
    accuracy -- Node giving the accuracy score on current batch
    x -- Node for input placeholder variable
    y -- Node for label placeholder variable
    """
    # I've started from a Keras model, rather than build a TensorFlow
    # one, but in order to get certain underlying TensorFlow nodes I
    # must compile it first, even if I don't use the Keras optimizer:
    keras_model = get_keras_model(use_dropout=use_dropout)
    model = tf.global_variables_initializer()
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    keras_model.compile(loss='categorical_crossentropy',
                        optimizer=sgd,
                        metrics=['accuracy'])
    accuracy = keras_model.metrics_tensors[0]
    x = keras_model.inputs[0]
    y = keras_model.targets[0]
    predict = keras_model.outputs[0]
    loss = tf.reduce_mean(categorical_crossentropy(y, predict))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return (model, train_op, accuracy, x, y)
