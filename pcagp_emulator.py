#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 2019

@author: cguilloteau

First emulator for a 1st galaxy training set (5 input parameters in GalSim :flux, radius, psf fwhm, shear profile g1 and g2)
"""

import numpy as np
from sklearn.decomposition import PCA
import h5py
# import pickle
import matplotlib.pyplot as plt
import GPy
from gengal import GenGalIm


def pca_reduction(X, ncomp=20):
    """
    Learn the PCA subspace from data X.

    Input parameters :
    - X : 2-D flatten data (nsamp * imsize)
    - ncomp : Dimension of the subspace

    Output :
    - pca : PCA model
    - weights : 2-D weights i.e. projection of the training set X onto the subspace spanned by the basis (nsamp * ncomp)
    """
    print('Performing dimensionality reduction ...')

    # PCA fitting
    pca = PCA(n_components=ncomp)
    weights = pca.fit_transform(X)
    basis = pca.components_

    # Plot cumsum(explained_variance) versus component
    plt.semilogy(pca.explained_variance_ratio_*100, 's')
    plt.ylabel('Explained Variance Ratio (%)', size=20)
    plt.xticks(size=20)
    plt.xlabel('Component', size=20)
    plt.yticks(size=20)
    plt.show()

    print('Explained variance ratio : '+str(round(np.cumsum(pca.explained_variance_ratio_)[-1]*100, 2))+' %.')

    # pickle.dump(pca, '/../Data/GPmodel/pca_'+str(ncomp))

    # Some plots on PCA
    plot_pca(basis, weights)

    return pca, weights


def plot_pca(basis, weights):
    """
    Makes some plots of basis and weights from PCA.
    Input :
    - basis : 2-D basis of the subspace (orthogonal vectors), (ncomp * imsize)
    - weights : 2-D weights i.e. projection of the training set X onto the subspace spanned by the basis (nsamp * ncomp)
    """
    file_name = '../Data/lhc_512_5.txt'
    params = np.loadtxt(file_name)

    ncomp, imsize = basis.shape
    npix = int(np.sqrt(imsize))
    ncol = int(ncomp//2)
    nsamp, _ = weights.shape

    # Shows the basis images
    for i in range(ncomp):
        plt.subplot(2, ncol, i+1)
        plt.imshow(basis[i].reshape((npix, npix)))
    plt.show()

    # Scatter plots
    for i in range(ncomp):
        # parameter number (0: flux, 1: radius, 2: g1 shear, 3: g2 shear, 4: psf fwhm)
        par = 3
        # weight number (x-axis) 0 -> ncomp-1
        w = 11
        plt.subplot(2, ncol, i+1)
        plt.scatter(weights[:, w], weights[:, i], s=1, c=params[:, par])
        plt.ylabel('Weight '+str(i+1), size=15)
        plt.xlabel('Weight '+str(w+1), size=15)
        plt.colorbar()
    plt.show()


def gp_fit(weights, params, task):
    """
    Learns the GP related to the weigths matrix
    Input :
    - weights : From PCA, 2-D weights i.e. projection of the training set X onto the subspace spanned by the basis (nsamp * ncomp)

    Output :
    - model : GP model
    - tmean, tmult : Rescaling factors
    """
    # Load and rescale parameters
    # file_name = '../Data/lhc_512_5.txt'
    # params = np.loadtxt(file_name)
    params, tmean, tmult = rescale(params)

    # Set the kernel
    # kernel = GPy.kern.Matern52(input_dim=params.shape[1], variance=.1, lengthscale=.1)
    kernel = GPy.kern.Matern52(input_dim=params.shape[1])

    # GP Regression
    model = GPy.models.GPRegression(params, weights, kernel=kernel)
    model.optimize()

    # Save model
    nparams = params.shape[1]
    ntrain = weights.shape[1]
    model.save_model('../Data/GPmodel/'+task+'gpfit_'+str(ntrain)+'_'+str(nparams), compress=True, save_data=True)
    return model, tmean, tmult


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


def emulator(pca, gp_model, params):
    """
    Emulates new images from physical parameters.

    Input :
    - pca : PCA model
    - gp_model : GP model
    - params : physical parameters (flux, radius, shear profile, psf fwhm)

    Output :
    - reconstructed : Emulated image
    """
    # Weights prediction
    pred_weights = gp_predict(gp_model, params)

    # Inverse PCA (pred_weights * basis + mean)
    reconstructed = pca.inverse_transform(pred_weights)
    return reconstructed


def rescale(params):
    """
    Rescales parameters between -1 and 1.

    Input :
    - params : physical parameters

    Output :
    - params_new : rescaled parameters
    - theta_mean, theta_mult : rescaling factors
    """
    theta_mean = np.mean(params, axis=0)
    theta_mult = np.max(params - theta_mean, axis=0)
    return (params - theta_mean) * theta_mult**-1, theta_mean, theta_mult


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
    # n_imgs, nx, ny = true.shape
    # true = np.reshape(true, (n_imgs, nx*ny))
    # predicted = np.reshape(predicted, (n_imgs, nx*ny))
    nx = 33
    ny = 33

    # Compute MSE
    se = np.sum((true - predicted)**2, axis=1)
    mse = se*(nx*ny)**-1

    # Compute R squared
    mean = np.mean(true, axis=1)
    r2 = 1 - se*np.sum((true - np.expand_dims(mean, axis=1))**2, axis=1)**-1

    return mse, r2


def perform_pca_gp(latent_dim, task, filename_train_gal, filename_train_par, filename_test_gal, filename_test_par):
    DataDir = "../Data/"
    # Load training set
    f = h5py.File('../Data/output_tests/'+filename_train_gal+'.hdf5', 'r')
    x_train = np.array(f['galaxies'])
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))

    # Load training parameters
    y_train = np.loadtxt(DataDir+filename_train_par)

    # PCA
    pca, W = pca_reduction(x_train, ncomp=latent_dim)
    # GP learning
    gp, tmean, tmult = gp_fit(W, y_train, task)

    # Load testing set
    f = h5py.File('../Data/output_tests/'+filename_test_gal+'.hdf5', 'r')
    x_test = np.array(f['galaxies'])
    x_test = (x_test - xmin) / xmax
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

    # Load training parameters
    y_test = np.loadtxt(DataDir+filename_test_par)

    # Predict x_test
    x_test_decoded = emulator(pca, gp, y_test)

    # Calculate mse
    mse, _ = mse_r2(x_test, x_test_decoded)

    np.savetxt('../Data/mse_test_'+task.txt, mse)

    return np.median(mse)


def main():
    # Load training set and rescale flux
    path = '../Data/output_cosmos/cosmos_real_train_512.hdf5'
    f = h5py.File(path, 'r')
    x_train = np.array(f['galaxies'])
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))

    path = '../Data/output_cosmos/cosmos_train_512.hdf5'
    f = h5py.File(path, 'r')
    x_train_par = np.array(f['galaxies'])
    xmin = np.min(x_train_par)
    xmax = np.max(x_train_par) - xmin
    x_train_par = (x_train_par - xmin) / xmax
    x_train_par = np.reshape(x_train_par, (x_train_par.shape[0], x_train_par.shape[1]*x_train_par.shape[2]))

    # New parameters to interpolate
    # params_new = np.zeros((10, 5))
    # for i in range(10):
    # params_new[i] = [np.random.uniform(1e4, 1e5), np.random.uniform(.1, 1.), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(.2, .4)]
    # DataDir = '../Data/'
    # y_train = np.loadtxt(DataDir + 'lhc_512_5.txt')
    # y_test = np.loadtxt(DataDir + 'lhc_64_5_testing.txt')

    # Perform PCA
    pca_real, W_real = pca_reduction(x_train, ncomp=20)
    pca_par, W_par = pca_reduction(x_train_par, ncomp=20)
    # GP learning
    # gp, tmean, tmult = gp_fit(W, y_train)

    # Rescale y_test
    # y_test = (y_test - tmean) * tmult**-1

    # Emulate
    # x_test_decoded = emulator(pca, gp, y_test)
    x_train_real_decoded = pca_real.inverse_transform(W_real)
    x_train_par_decoded = pca_par.inverse_transform(W_par)

    # GalSim images simulation
    nx = 64  # pixels in the 1st spatial dimension
    ny = 64  # pixels in the 2nd spatial dimension
    # Initialize bunch array
    # x_test = np.zeros((params_new.shape[0], nx, ny))
    # for i in range(params_new.shape[0]):
    # x_test[i] = GenGalIm(params_new[i]).array
    # x_test = np.array(h5py.File(DataDir + '/output_tests/test_64_5_testing.hdf5', 'r')['galaxies'])
    # x_test = (x_test - xmin) / xmax

    # Save training/testing sets
    # np.savetxt(DataDir+'pca_decoded_xtest_64_5.txt', x_test_decoded)
    # x_test = np.reshape(x_test, (64, 33*33))

    # Compute mse and r2
    # mse_train_pca, r2_train_pca = mse_r2(x_train, x_train_decoded)
    # mse_test_pca, r2_test_pca = mse_r2(x_test, x_test_decoded)

    # # Plot Xnew vs GalSim
    n = 167

    for i in range(10):
        plt.subplot(4, 10, i+1)
        plt.imshow(np.reshape(x_train_par[i+n], (nx, ny)))

        plt.subplot(4, 10, 10+i+1)
        plt.imshow(np.reshape(x_train[i+n], (nx, ny)))
        # plt.title('Emulated image using PCA + GP '+str(i))
        # plt.colorbar()
        plt.subplot(4, 10, 20+i+1)
        plt.imshow(np.reshape(x_train_par_decoded[i+n], (nx, ny)))
        # plt.title('Simulated image using GalSim '+str(i))
        # plt.colorbar()
        plt.subplot(4, 10, 30+i+1)
        plt.imshow(abs(np.reshape(x_train_real_decoded[i+n], (nx, ny))))
        # mse = np.mean((np.reshape(x_test_decoded[i], (nx, ny))-x_test[i])**2)
        # plt.title('MSE = '+str(mse), size=10)
        # plt.subplot(5, 10, 40+i+1)
        # plt.imshow(abs(np.reshape(x_train_par[i+n], (nx, ny))-np.reshape(x_train[i+n], (nx, ny))))

    plt.show()

    # return mse_train_pca, mse_test_pca, r2_train_pca, r2_test_pca
    # return mse_train_pca, r2_train_pca