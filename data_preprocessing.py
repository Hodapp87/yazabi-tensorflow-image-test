#!/usr/bin/env python

###########################################################################
# data_preprocessing.py: Data preprocessing for image recognition test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-28
###########################################################################

import scipy.io
import numpy as np
import sklearn.model_selection

def normalize(images):
    """Normalizes input RGB images (i.e. zero mean & unit variance for
    each pixel coordinate); returns a new array of images with shape
    (N,Y,X,3).
    
    Parameters:
    images -- Input array, shape (N,Y,X,3), for N color XxY images
    """
    mean = images.mean(axis=0)
    stdev = images.std(axis=0)
    img_norm = (images - mean) / stdev
    return img_norm

def normalize_greyscale(images):
    """Converts input RGB images to greyscale, and normalizes them
    (i.e. zero mean & unit variance for each pixel coordinate);
    returns a new array of images with shape (N,Y,X).
    
    Parameters:
    images -- Input array, shape (N,Y,X,3), for N color XxY images

    """
    r, g, b = images[:,:,:,0], images[:,:,:,1], images[:,:,:,2]
    # Standard NTSC/PAL luminance:
    img_lum = 0.2989*r + 0.5870*g + 0.1140*b
    # Note the axis=0. This normalization is per-pixel (though since
    # every pixel has been normalized, the aggregate mean & stdev are
    # also 0 and 1)
    mean = img_lum.mean(axis=0)
    stdev = img_lum.std(axis=0)
    img_norm = (img_lum - mean) / stdev
    return img_norm

def split(train_X_orig, train_y_orig, ratio=0.75):
    """Splits original training data (i.e. what is supplied in
    'train_32x32.mat' into training and validation data, using 'ratio'
    as the amount that should be split for training (default 0.7).
    Returns (train_X, validation_X, train_y, validation_y).
    """
    # Class labels are a bit lopsided, so stratified sampling is done:
    splits = sklearn.model_selection.train_test_split(
        train_X_orig,
        train_y_orig,
        test_size=1 - ratio,
        random_state=123456,
        stratify=train_y_orig)
    return splits

def load_data():
    """Loads data from the SVHN (Street View House Numbers) dataset.  This
    assumes that train_32x32.mat and test_32x32.mat have been
    downloaded into the 'data' directory.

    Returns:

    (train_X, train_y, test_X, test_y) where train_X/test_X are
    4-dimensional arrays of shape (N, 32, 32, 3), representing N color
    32x32 images, and train_y/test_y are N-element arrays giving the
    corresponding number as an integer (0-9).
    """
    train = scipy.io.loadmat("data/train_32x32.mat")
    test = scipy.io.loadmat("data/test_32x32.mat")
    # transpose is to put these closer to something like a design
    # matrix, where each row contains a sample (though 'row' here is
    # multidimensional):
    train_X = np.transpose(train["X"], (3, 0, 1, 2))
    test_X = np.transpose(test["X"], (3, 0, 1, 2))
    # y is given as a (N,1) array; squeeze drops the final 1:
    train_y = train["y"].squeeze()
    test_y = test["y"].squeeze()
    # A label of 10 is used for a digit of 0, so simplify this:
    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0
    return (train_X, train_y, test_X, test_y)
