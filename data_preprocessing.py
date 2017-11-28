#!/usr/bin/env python

###########################################################################
# data_preprocessing.py: Data preprocessing for image recognition test
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2017-11-28
###########################################################################

import scipy.io
import numpy as np

def normalize(images):
    """Converts input images to greyscale, and normalizes them
    (i.e. zero mean & unit variance); returns a new array of images
    with shape (Y,X,N).
    
    Parameters:
    images -- Input array, shape (Y,X,3,N), for N color XxY images
    """
    r, g, b = images[:,:,0,:], images[:,:,1,:], images[:,:,2,:]
    img_lum = 0.2989*r + 0.5870*g + 0.1140*b
    img_norm = (img_lum - img_lum.mean()) / img_lum.std()
    return img_norm

def load_data():
    """Loads data from the SVHN (Street View House Numbers) dataset.  This
    assumes that train_32x32.mat and test_32x32.mat have been
    downloaded into the 'data' directory.

    Returns:

    (train_X, train_y, test_X, test_y) where train_X/test_X are
    4-dimensional arrays of shape (32, 32, 3, N), representing N color
    32x32 images, and train_y/test_y are N-element arrays giving the
    corresponding number as an integer.
    """
    train = scipy.io.loadmat("data/train_32x32.mat")
    test = scipy.io.loadmat("data/test_32x32.mat")
    # y is given as a (N,1) array; squeeze drops the final 1
    train_X, train_y = train["X"], train["y"].squeeze()
    test_X,  test_y  = test["X"],  test["y"].squeeze()
    return (train_X, train_y, test_X, test_y)
