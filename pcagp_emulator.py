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
    file_name = '../Data/lhc_512_5_lownoise.txt'
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


def gp_fit(weights):
    """
    Learns the GP related to the weigths matrix
    Input :
    - weights : From PCA, 2-D weights i.e. projection of the training set X onto the subspace spanned by the basis (nsamp * ncomp)

    Output :
    - model : GP model
    - tmean, tmult : Rescaling factors
    """
    # Load and rescale parameters
    file_name = '../Data/lhc_512_5_lownoise.txt'
    params = np.loadtxt(file_name)
    params, tmean, tmult = rescale(params)

    # Set the kernel
    # kernel = GPy.kern.Matern52(input_dim=params.shape[1], variance=.1, lengthscale=.1)
    kernel = GPy.kern.Matern52(input_dim=params.shape[1])

    # GP Regression
    model = GPy.models.GPRegression(params, weights, kernel=kernel)
    model.optimize()

    # Save model
    model.save_model('../Data/GPmodel/gpfit_lownoise', compress=True, save_data=True)
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


# Load training set and rescale flux
path = '../Data/output_tests/training_512_5_lownoise.hdf5'
f = h5py.File(path, 'r')
X = np.array(f['galaxies'])
xmax = np.max(X)
X /= xmax
X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))

# New parameters to interpolate
params_new = np.zeros((10, 5))
for i in range(10):
    params_new[i] = [np.random.uniform(1e4, 1e5), np.random.uniform(.1, 1.), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(.2, .4)]

# Perform PCA
pca, W = pca_reduction(X, ncomp=14)
# GP learning
gp, tmean, tmult = gp_fit(W)

# Rescale new parameters
params_new_rescale = (params_new - tmean) * tmult**-1

# Emulate
Xnew_em = emulator(pca, gp, params_new_rescale)

# GalSim images simulation
nx = 33  # pixels in the 1st spatial dimension
ny = 33  # pixels in the 2nd spatial dimension
# Initialize bunch array
Xnew_gs = np.zeros((params_new.shape[0], nx, ny))
for i in range(params_new.shape[0]):
    Xnew_gs[i] = GenGalIm(params_new[i]).array

# Plot Xnew vs GalSim
for i in range(10):
    plt.subplot(3, 10, i+1)
    plt.imshow(np.reshape(Xnew_em[i]*xmax, (nx, ny)))
    # plt.title('Emulated image using PCA + GP '+str(i))
    # plt.colorbar()
    plt.subplot(3, 10, 10+i+1)
    plt.imshow(Xnew_gs[i])
    # plt.title('Simulated image using GalSim '+str(i))
    # plt.colorbar()
    plt.subplot(3, 10, 20+i+1)
    plt.imshow(abs(np.reshape(Xnew_em[i]*xmax, (nx, ny))-Xnew_gs[i]))
    mse = np.mean((np.reshape(Xnew_em[i]*xmax, (nx, ny))-Xnew_gs[i])**2)
    plt.title('MSE = '+str(mse), size=10)
plt.show()
