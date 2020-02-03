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
from matplotlib.colors import LogNorm
import galsim
import os
import umap


def pca_reduction(X, y_train, ncomp=20):
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
    # plt.semilogy(np.cumsum(pca.explained_variance_ratio_*100), 's')
    # plt.ylabel('Explained Variance Ratio (%)', size=20)
    # plt.xticks(size=20)
    # plt.xlabel('Component', size=20)
    # plt.yticks(size=20)
    # plt.show()

    print('Explained variance ratio : '+str(round(np.cumsum(pca.explained_variance_ratio_)[-1]*100, 2))+' %.')

    # pickle.dump(pca, '/../Data/GPmodel/pca_'+str(ncomp))

    # Some plots on PCA
    plot_pca(basis, weights, y_train)

    return pca, weights


def rescale_im(img):
    """
    Rescale an image between 0 and 1 to plot.
    """
    return (img - np.min(img)) / np.max(img - np.min(img))


def plot_pca(basis, weights, y):
    """
    Makes some plots of basis and weights from PCA.
    Input :
    - basis : 2-D basis of the subspace (orthogonal vectors), (ncomp * imsize)
    - weights : 2-D weights i.e. projection of the training set X onto the subspace spanned by the basis (nsamp * ncomp)
    """
    # file_name = '../Data/Galsim/lhc_512_5.txt'
    # params = np.loadtxt(file_name)
    params = y

    # ncomp, imsize = basis.shape
    # npix = int(np.sqrt(imsize))
    # basis = np.array(np.reshape(basis, (ncomp, npix, npix)))
    # ncol = int(ncomp//2)
    # nsamp, _ = weights.shape

    # basis_plot = rescale_im(basis[0])
    # for i in range(ncol-1):
    #     basis_plot = np.concatenate((basis_plot, rescale_im(basis[i+1])), axis=1)
    # basis_plot_ = rescale_im(basis[ncol])
    # for i in range(ncol-1):
    #     basis_plot_ = np.concatenate((basis_plot_, rescale_im(basis[ncol+i+1])), axis=1)
    # basis_plot = np.concatenate((basis_plot, basis_plot_), axis=0)
    # # Shows the basis images
    # plt.imshow(basis_plot, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # parameter number (0: flux, 1: radius, 2: g1 shear, 3: g2 shear, 4: psf fwhm)
    par = 2
    # weight number (x-axis) 0 -> ncomp-1
    w1 = 2
    w2 = 3
    plt.scatter(weights[:, w1], weights[:, w2], c=params[:, par])
    plt.ylabel('Weight dimension '+str(w2+1), size=15)
    plt.xlabel('Weight dimension'+str(w1+1), size=15)
    plt.colorbar()
    plt.show()

    reducer = umap.UMAP()
    embedding_train = reducer.fit_transform(weights)
    # embedding_test = reducer.transform(x_test_encoded)
    plt.figure()
    plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=params[:, par].flatten(), cmap='bone', norm=LogNorm())
    # plt.colorbar()
    # plt.scatter(embedding_test[:, 0], embedding_test[:, 1], c=y_test[:, 0].flatten(), cmap='Wistia', norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    # plt.savefig(PlotDir+'cosmos_umap_flux.png', figsize=(20000, 20000), bbox_inches="tight")
    # plt.close()


def gp_fit(weights, params, task):
    """
    Learns the GP related to the weigths matrix
    Input :
    - weights : From PCA, 2-D weights i.e. projection of the training set X onto the subspace spanned by the basis (nsamp * ncomp)

    Output :
    - model : GP model
    - tmean, tmult : Rescaling factors
    """
    print('GP training ...')
    # Load and rescale parameters
    # file_name = '../Data/lhc_512_5.txt'
    # params = np.loadtxt(file_name)
    params, tmean, tmult = rescale(params)

    # Set the kernel
    # kernel = GPy.kern.Matern52(input_dim=params.shape[1], variance=.01, lengthscale=.1)
    kernel = GPy.kern.Matern52(input_dim=params.shape[1])

    # GP Regression
    model = GPy.models.GPRegression(params, weights, kernel=kernel)
    model.optimize()

    # Save model
    nparams = params.shape[1]
    ntrain = weights.shape[1]
    model.save_model('../Data/GPmodel/'+task+'gpfit_'+str(ntrain)+'_'+str(nparams), compress=True, save_data=True)
    print('end GP training ...')
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


def shear_estimation(PlotDir, true, predicted):
    """
    Using galsim.hsm.EstimateShear(gal_img, psf) : Estimate galaxy shear, correcting for the conv. by psf.
    """
    big_run_params = galsim.hsm.HSMParams(max_mom2_iter=40000)
    n_imgs = true.shape[0]

    shear_true = np.zeros((n_imgs, 2))
    shear_pred = np.zeros((n_imgs, 2))

    for i in range(n_imgs):
        # shape_true = galsim.hsm.EstimateShear(galsim.Image(true[i]), galsim.Image(psf[i])).observed_shape
        # shape_pred = galsim.hsm.EstimateShear(galsim.Image(predicted[i]), galsim.Image(psf[i])).observed_shape
        print(i)
        if i == 280:
            i += 1

        psf = galsim.Gaussian(fwhm=0.2)

        shape_true = galsim.hsm.EstimateShear(galsim.Image(true[i]), psf.drawImage(nx=32, ny=32, scale=0.2), hsmparams=big_run_params).observed_shape
        shape_pred = galsim.hsm.EstimateShear(galsim.Image(predicted[i]), psf.drawImage(nx=32, ny=32, scale=0.2), hsmparams=big_run_params).observed_shape

        shear_true[i] = np.array([shape_true.g1, shape_true.g2])
        shear_pred[i] = np.array([shape_pred.g1, shape_pred.g2])

    countsg1_true, bins1 = np.histogram(shear_true[:, 0], bins=25)
    countsg1_pred, bins1 = np.histogram(shear_pred[:, 0], bins=bins1)
    countsg2_true, bins2 = np.histogram(shear_true[:, 1], bins=25)
    countsg2_pred, bins2 = np.histogram(shear_pred[:, 1], bins=bins2)
    counts_true, bins3 = np.histogram(shear_true[:, 0]**2 + shear_true[:, 1]**2, bins=25)
    counts_pred, bins3 = np.histogram(shear_pred[:, 0]**2 + shear_pred[:, 1]**2, bins=bins3)

    plt.figure('Shear estimation')
    plt.subplot(221)
    plt.semilogy(bins1[:-1], countsg1_true, 's--', label='True g1')
    plt.semilogy(bins1[:-1], countsg1_pred, 's--', label='Predicted g1')
    plt.legend()
    plt.xlabel('g1 value')
    plt.ylabel('Counts')
    plt.subplot(222)
    plt.semilogy(bins2[:-1], countsg2_true, 's--', label='True g2')
    plt.semilogy(bins2[:-1], countsg2_pred, 's--', label='Predicted g2')
    plt.legend()
    plt.xlabel('g2 value')
    plt.ylabel('Counts')
    plt.subplot(223)
    plt.scatter(shear_true[:, 0], shear_pred[:, 0], s=1)
    plt.xlabel('True g1')
    plt.ylabel('Predicted g1')
    plt.legend()
    plt.subplot(224)
    # diff_module = abs((shear_true[:, 0]**2 + shear_true[:, 1]**2) - (shear_pred[:, 0]**2 + shear_pred[:, 1]**2)) * ((shear_true[:, 0]**2 + shear_true[:, 1]**2) **(-1)) * 100
    # plt.plot(np.sort(diff_module), 's--', label='g1$^2$ + g2$^2$')
    # plt.plot(np.sort(diff_g1), 's--', label='g1')
    # plt.plot(np.sort(diff_g2), 's--', label='g2')
    # plt.xlabel('g1$^2$ + g2$^2$')
    # plt.ylabel('Error ($\%$)')
    plt.scatter(shear_true[:, 1], shear_pred[:, 1], s=1)
    plt.xlabel('True g2')
    plt.ylabel('Predicted g2')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(PlotDir+'pcagp_shear_estimation.png')
    plt.close()


def pixel_intensity(PlotDir, true, predicted):
    """
    Plots the distribution of pixel intensities for the validation set (or any set made of real/input image set) and the reconstructed/predicted data set.
    """
    counts_true, bins = np.histogram(true.flatten(), bins=100)
    counts_pred, bins = np.histogram(predicted.flatten(), bins=bins)

    plt.figure('Pixel flux value distribution')
    plt.subplot(121)
    plt.semilogy(bins[:-1], counts_true, 's-', label='True')
    plt.semilogy(bins[:-1], counts_pred, 's-', label='Predicted')
    plt.legend()
    plt.xlabel('Pixel intensity')
    plt.ylabel('Counts')
    # plt.show()
    plt.subplot(122)
    plt.scatter(true.flatten(), predicted.flatten(), s=1)
    plt.plot(np.linspace(0, 0.175), np.linspace(0, 0.175), 'r')
    plt.xlabel('True pixel intensity')
    plt.ylabel('Predicted pixel intensity')
    plt.tight_layout()
    plt.savefig(PlotDir+'pca_pixels_intensity.png')
    plt.close()


def main():
    n_train = 4096
    n_test = 256
    nx = 64
    ny = 64

    # ------------------------------ LOAD DATA ----------------------------------
    # Load training set images and rescale fluxes
    path = '../Data/Cosmos/data/cosmos_real_trainingset_train_'+str(n_train)+'_test_'+str(n_test)+'.hdf5'
    f = h5py.File(path, 'r')
    x_train = np.array(f['real galaxies'])
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_train = np.reshape(x_train, (n_train, nx*ny))

    # Load testing set parametric images
    # x_train_parametric = (np.array(f['parametric galaxies']) - xmin) / xmax
    # x_train_parametric = np.reshape(x_train_parametric, (n_train, nx*ny))

    # Load training set parameters and rescale
    y_train = np.array(f['parameters'])
    f.close()

    # Load testing set and rescale fluxes
    path = '../Data/Cosmos/data/cosmos_real_testingset_train_'+str(n_train)+'_test_'+str(n_test)+'.hdf5'
    f = h5py.File(path, 'r')
    x_test = np.array(f['real galaxies'])
    x_test = (x_test - xmin) / xmax
    x_test = np.reshape(x_test, (n_test, nx*ny))

    # Load testing set parameters and rescale
    y_test = np.array(f['parameters'])

    # Load testing set parametric images
    # x_test_parametric = (np.array(f['parametric galaxies']) - xmin) / xmax
    # x_test_parametric = np.reshape(x_test_parametric, (n_test, nx*ny))
    f.close()

    # ----------------------- PERFORM PCA+GP TRAINING ---------------------------
    # Perform PCA
    pca, W = pca_reduction(x_train, y_train, ncomp=20)
    # GP learning
    # gp, tmean, tmult = gp_fit(W, y_train[:, :4], 'cosmos')

    # Rescale y_test
    # y_test[:, :4] = (y_test[:, :4] - tmean) * tmult**-1
    # y_train[:, :4] = (y_train[:, :4] - tmean) * tmult**-1

    # ----------------------- EMULATION -----------------------------------------
    # Emulate
    # x_test_gp_decoded = np.reshape(emulator(pca, gp, y_test[:, :4]), (n_test, nx, ny))
    # x_train_gp_decoded = emulator(pca, gp, y_train[:, :4])

    # Generate decoded training set
    # x_train_decoded_pca = pca.inverse_transform(W)

    x_test_decoded = np.reshape(pca.inverse_transform(pca.transform(x_test)), (n_test, nx, ny))

    x_test = np.reshape(x_test, (n_test, nx, ny))

    # print('Shear estimation ...')
    # shear_estimation('../Plots/Cosmos_plots/', x_test, x_test_decoded)

    # print('Pixel intensity ...')
    # pixel_intensity('../Plots/Cosmos_plots/', x_test, x_test_decoded)

    # print('Plotting ...')
    # plt.figure()
    # orig_plots = rescale_im(x_test[0])
    # for i in range(9):
    #     orig_plots = np.concatenate((orig_plots, rescale_im(x_test[i+2])), axis=1)

    # for i in range(1):
    #     orig_plots_ = rescale_im(x_test[10*(i+1)+2])
    #     for j in range(9):
    #         orig_plots_ = np.concatenate((orig_plots_, rescale_im(x_test[10*(i+1)+2+j])), axis=1)
    #     orig_plots = np.concatenate((orig_plots, orig_plots_), axis=0)

    # plt.imshow(orig_plots)
    # plt.axis('off')
    # # plt.show()

    # plt.figure()
    # rec_plots = rescale_im(x_test_decoded[0])
    # for i in range(9):
    #     rec_plots = np.concatenate((rec_plots, rescale_im(x_test_decoded[i+2])), axis=1)

    # for i in range(1):
    #     rec_plots_ = rescale_im(x_test_decoded[10*(i+1)+2])
    #     for j in range(9):
    #         rec_plots_ = np.concatenate((rec_plots_, rescale_im(x_test_decoded[10*(i+1)+2+j])), axis=1)
    #     rec_plots = np.concatenate((rec_plots, rec_plots_), axis=0)

    # plt.imshow(rec_plots)
    # plt.axis('off')
    # # plt.show()

    # x_test_decoded_noise = x_test_decoded + 4e-4*np.random.randn(x_test_decoded.shape[0], x_test_decoded.shape[1], x_test_decoded.shape[2])

    # plt.figure()
    # rec_plots = rescale_im(x_test_decoded_noise[0])
    # for i in range(9):
    #     rec_plots = np.concatenate((rec_plots, rescale_im(x_test_decoded_noise[i+2])), axis=1)

    # for i in range(1):
    #     rec_plots_ = rescale_im(x_test_decoded_noise[10*(i+1)+2])
    #     for j in range(9):
    #         rec_plots_ = np.concatenate((rec_plots_, rescale_im(x_test_decoded_noise[10*(i+1)+2+j])), axis=1)
    #     rec_plots = np.concatenate((rec_plots, rec_plots_), axis=0)
    # plt.imshow(rec_plots)
    # plt.axis('off')
    # plt.show()

    # GalSim images simulation
    # # Initialize bunch array
    # x_test = np.zeros((y_test.shape[0], nx, ny))
    # for i in range(y_test.shape[0]):
    #     x_test[i] = GenGalIm(y_test[i]).array
    #     # x_test = np.array(h5py.File(DataDir + '/output_tests/test_64_5_testing.hdf5', 'r')['galaxies'])
    #     x_test = (x_test - xmin) / xmax

    # Save training/testing sets
    # np.savetxt(DataDir+'pca_decoded_xtest_64_5.txt', x_test_decoded)
    # x_test = np.reshape(x_test, (64, 33*33))

    # Compute mse and r2
    # mse_train_pca, r2_train_pca = mse_r2(x_train, x_train_decoded)
    # mse_test_pca, r2_test_pca = mse_r2(x_test, x_test_decoded)

    # # Plot Xnew vs GalSim

    # for i in range(10):
    #     plt.subplot(4, 10, i+1)
    #     plt.imshow(np.reshape(x_train_parametric[i+n], (nx, ny)))

    #     plt.subplot(4, 10, 10+i+1)
    #     plt.imshow(np.reshape(x_train[i+n], (nx, ny)))
    #     # plt.title('Emulated image using PCA + GP '+str(i))
    #     # plt.colorbar()
    #     plt.subplot(4, 10, 20+i+1)
    #     plt.imshow(np.reshape(x_train_decoded_pca[i+n], (nx, ny)))
    #     # plt.title('Simulated image using GalSim '+str(i))
    #     # plt.colorbar()
    #     plt.subplot(4, 10, 30+i+1)
    #     plt.imshow(np.reshape(x_train_decoded[i+n], (nx, ny)))
    #     # plt.subplot(4, 10, 30+i+1)
        # plt.imshow(abs(np.reshape(x_train_real_decoded[i+n], (nx, ny))))
        # mse = np.mean((np.reshape(x_test_decoded[i], (nx, ny))-x_test[i])**2)
        # plt.title('MSE = '+str(mse), size=10)
        # plt.subplot(4, 10, 30+i+1)
        # plt.imshow(abs(np.reshape(x_test[i+n], (nx, ny))-np.reshape(x_test_decoded[i+n], (nx, ny))))

    # plt.show()

    # return mse_train_pca, mse_test_pca, r2_train_pca, r2_test_pca
    # return mse_train_pca, r2_train_pca
