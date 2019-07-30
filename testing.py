#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 2019

@author: cguilloteau
"""

import numpy as np
import network_params as netparam
from keras.models import load_model
import h5py
from keras import backend as K


def mse_r2(true, predicted):
    """
    Compute the mean square error (mse) and the r squared error (r2) of the predicted set of images.
    Inputs :
    - true : the original simulated set of images (n_imgs, nx, ny)
    - predicted : reconstructed set of images (n_imgs, nx, ny)
    Outputs :
    - mse : Mean Square Error
    - r2 : R-squared Error
    """
    # Reshaping set of images
    n_imgs, nx, ny = true.shape
    true = np.reshape(true, (n_imgs, nx*ny))
    predicted = np.reshape(predicted, (n_imgs, nx*ny))

    # Compute MSE
    se = np.sum((true - predicted)**2, axis=1)
    mse = se*(nx*ny)**-1

    # Compute R squared
    mean = np.mean(true, axis=1)
    r2 = 1 - se*np.sum((true - np.expand_dims(mean, axis=1))**2, axis=1)**-1

    return mse, r2


def compute_vae_mse(filename_decoder, filename_test_encoded, filename_train_gal, filename_test_gal):

    # ------------------------ Parameters ---------------------------------------
    DataDir = netparam.DataDir

    # ------------------------ Load data and models -----------------------------

    # Load training and testing set
    x_train = np.array(h5py.File(DataDir + 'output_tests/'+filename_train_gal+'.hdf5', 'r')['galaxies'])[:, :32, :32]
    x_test = np.array(h5py.File(DataDir + 'output_tests/'+filename_test_gal+'.hdf5', 'r')['galaxies'])[:, :32, :32]

    # Rescaling
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_test = (x_test - xmin) / xmax
    x_train = K.cast_to_floatx(x_train)
    x_test = K.cast_to_floatx(x_test)

    # Compute reconstructed testing set
    x_test_encoded = np.loadtxt(DataDir+filename_test_encoded)
    x_test_encoded = K.cast_to_floatx(x_test_encoded)
    x_test_encoded = x_test_encoded.astype('float32')

    decoder = load_model(DataDir+filename_decoder)
    # x_test_decoded = np.zeros((ntest, nx, ny))
    x_test_decoded = decoder.predict(x_test_encoded)

    # Compute MSE and R_squared
    mse, _ = mse_r2(x_test, x_test_decoded)

    return np.median(mse)


def main():
    # galsim.hsm.EstimateShear ? But need an estimate of the PSF

    # ------------------------ Parameters ---------------------------------------
    DataDir = netparam.DataDir
    ntrain = netparam.ntrain
    ntest = netparam.ntest
    nx = netparam.nx
    ny = netparam.ny

    # ------------------------ Load data and models -----------------------------

    # Load training and testing set
    x_train = np.array(h5py.File(DataDir + '/output_tests/training_512_5.hdf5', 'r')['galaxies'])[:, :32, :32]
    x_test = np.array(h5py.File(DataDir + '/output_tests/test_64_5_testing.hdf5', 'r')['galaxies'])[:, :32, :32]

    # Rescaling
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_test = (x_test - xmin) / xmax
    x_train = K.cast_to_floatx(x_train)
    x_test = K.cast_to_floatx(x_test)

    # Compute reconstructed testing set
    x_test_encoded = np.loadtxt(DataDir+'cvae_encoded_xtest_512_5.txt')
    x_test_encoded = K.cast_to_floatx(x_test_encoded)
    x_test_encoded = x_test_encoded.astype('float32')

    decoder = load_model(DataDir+'cvae_decoder_model_galsim.h5')
    # x_test_decoded = np.zeros((ntest, nx, ny))
    x_test_decoded = decoder.predict(x_test_encoded)

    # Load reconstructed training set
    x_train_decoded = np.reshape(np.loadtxt(DataDir+'cvae_decoded_xtrain_512_5.txt'), (ntrain, nx, ny))

    # Compute MSE and R_squared
    mse_train, r2_train = mse_r2(x_train, x_train_decoded)
    mse_test, r2_test = mse_r2(x_test, x_test_decoded)

    return mse_train, mse_test, r2_train, r2_test
