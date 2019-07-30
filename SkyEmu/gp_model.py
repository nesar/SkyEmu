#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 2019

@author: cguilloteau
"""

import numpy as np
# import matplotlib.pyplot as plt
import GPy
import network_params as netparam


def rescale(params):
    """
    Rescales parameters between -1 and 1.
    Input :
    - params : physical parameters
    Outputs :
    - params_new : rescaled parameters
    - theta_mean, theta_mult : rescaling factors
    """
    theta_mean = np.mean(params, axis=0)
    theta_mult = np.max(params - theta_mean, axis=0)
    return (params - theta_mean) * theta_mult**-1, theta_mean, theta_mult


# ----------------------- GP fitting and predicting functions ----------------


def gp_fit(weights, y_train, task):
    """
    Learns the GP related to the weigths matrix
    Input :
    - weights : From encoder (2-D) : x_train_encoded
    - y_train : Physical parameters to interpolate

    Output :
    - model : GP model
    """
    DataDir = netparam.DataDir
    # Set the kernel
    # kernel = GPy.kern.Matern52(input_dim=params.shape[1], variance=.1, lengthscale=.1)
    kernel = GPy.kern.Matern52(input_dim=y_train.shape[1])

    # GP Regression
    model = GPy.models.GPRegression(y_train, weights, kernel=kernel)
    model.optimize()

    # Save model
    nparams = y_train.shape[1]
    ntrain = weights.shape[1]
<<<<<<< HEAD:gp_model.py
    model.save_model(DataDir'GPmodel/'+task+'gpfit_cvae_'+str(ntrain)+'_'+str(nparams), compress=True, save_data=True)
=======
    model.save_model(DataDir+task+'GPmodel/gpfit_cvae_'+str(ntrain)+'_'+str(nparams), compress=True, save_data=True)
>>>>>>> 32dc5195e73758775c80c447f4534535837a4a3b:SkyEmu/gp_model.py
    return model


def gp_predict(model, params):
    """
    Predicts the weights matrix to feed inverse PCA from physical parameters.

    Input :
    - model : GP model
    - params : physical parameters (flux, radius, shear profile, psf fwhm)

    Output :
    - predic[0] : predicted weights
    """
    predic = model.predict(params)
    return predic[0]


def gp(task, filename_train_par, filename_test_par, filename_train_encoded):
    # ------------------------ Parameters ---------------------------------------

    DataDir = netparam.DataDir

    # ------------------------ Loading and rescaling ----------------------------

    y_train = np.loadtxt(DataDir + filename_train_par)
    ntrain = y_train.shape[0]
    y_test = np.loadtxt(DataDir + filename_test_par)

    y_train, ymean, ymult = rescale(y_train)
    y_test = (y_test - ymean) * ymult**-1

    x_train_encoded = np.loadtxt(DataDir + filename_train_encoded)

    # ---------------------- Training -------------------------------------------

<<<<<<< HEAD:gp_model.py
    gpmodel = gp_fit(x_train_encoded, y_train, task)
    x_test_encoded = gp_predict(gpmodel, y_test)

    filename_test_encoded = task + 'cvae_encoded_xtest_5_'+str(ntrain)+'.txt'
=======
    gpmodel = gp_fit(task, x_train_encoded, y_train)
    x_test_encoded = gp_predict(gpmodel, y_test)

    filename_test_encoded = task+ 'cvae_encoded_xtest_5_'+str(ntrain)+'.txt'
>>>>>>> 32dc5195e73758775c80c447f4534535837a4a3b:SkyEmu/gp_model.py
    np.savetxt(DataDir+filename_test_encoded, x_test_encoded)

    return filename_test_encoded


def main():
    # ------------------------ Parameters ---------------------------------------

    DataDir = netparam.DataDir

    # ------------------------ Loading and rescaling ----------------------------

    y_train = np.loadtxt(DataDir + 'lhc_512_5.txt')
    y_test = np.loadtxt(DataDir + 'lhc_64_5_testing.txt')

    y_train, ymean, ymult = rescale(y_train)
    y_test = (y_test - ymean) * ymult**-1

    x_train_encoded = np.loadtxt(DataDir+'cvae_encoded_xtrain_512_5.txt')

    # ---------------------- Training -------------------------------------------

    gpmodel = gp_fit(x_train_encoded, y_train)
    x_test_encoded = gp_predict(gpmodel, y_test)
    np.savetxt(DataDir+'cvae_encoded_xtest_512_5.txt', x_test_encoded)
