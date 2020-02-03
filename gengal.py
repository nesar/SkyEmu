#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 2019

@author: cguilloteau

Simple function that, given input parameters (flux, radius, psf fwhm, shear profile), returns an image of a galaxy without noise using GalSim.
"""

import galsim
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from astropy.io import fits
# from time import time


def GenGalIm(params):
    """
    Input parameters :

    params : Array with :
    - flux : Flux of the galaxy in counts.
    - radius : Radius of the galaxy in arcsec.
    - g1 : Reduced shear component.
    - g2 : Reduced shear component. See demo2 comments. sqrt(g1**2+g2**2) < 1.0.
    - psf_fwhm : fwhm of the Gaussian PSF used for convolution.

    Output :
    - image : 33x33 pixels image of a galaxy given input parameters.
    """

    # Parameters
    galflux, galradius, g1, g2, psffwhm = params

    # Fixed parameters
    nx = 33  # pixels in the 1st spatial dimension
    ny = 33  # pixels in the 2nd spatial dimension
    pixel_scale = 0.2  # arcsec/pixel

    # t1 = time()

    # Define the galaxy profile
    gal = galsim.Exponential(flux=galflux, scale_radius=galradius)
    gal = gal.shear(g1=g1, g2=g2)

    # Define the PSF profile
    psf = galsim.Gaussian(fwhm=psffwhm)

    # Convolution
    big_fft_params = galsim.GSParams(maximum_fft_size=12300)
    final = galsim.Convolve([gal, psf], gsparams=big_fft_params)

    # Draw the image
    image = final.drawImage(nx=nx, ny=ny, scale=pixel_scale)

    # Add Poisson noise eventually
    # rng = galsim.BaseDeviate(150000)
    # noise = galsim.PoissonNoise(rng)
    # image.addNoise(noise)

    # t2 = time()
    # print('Time : '+str(t2-t1))

    return image


# def GenSetGal(file_name):
def GenSetGal(params):
    """
    Given a file name containing parameters, generates a bunch of galaxy images.
    """
    # Load parameters
    # params = np.loadtxt(file_name)
    # Fixed parameters
    nx = 33  # pixels in the 1st spatial dimension
    ny = 33  # pixels in the 2nd spatial dimension

    # Initialize bunch array
    setgal = np.zeros((params.shape[0], nx, ny))

    for i in range(params.shape[0]):
        setgal[i] = GenGalIm(params[i]).array

    return setgal


def SaveGal(images, fname, dname):
    """
    Save a single galaxy image into a hdf5 file.
    Input parameters :

    - images : Galaxy images to save
    - fname : file name
    - dname : dataset name
    """
    if not os.path.isdir('../Data/output_tests'):
        os.mkdir('output_tests')
    file_name = os.path.join('../Data/output_tests', fname+'.hdf5')
    f = h5py.File(file_name, 'w')
    f.create_dataset(dname, data=images)
    f.close()

    hdu = fits.PrimaryHDU(images)
    hdu.writeto('/Users/claireguilloteau/Documents/KSPA2019/SkyEmu/Data/output_tests/'+fname+'.fits', overwrite=True)


def PlotGal():
    """
    Given parameters (flux, radius, psf fwhm, shear profile), plots the related galaxy image.

    Test function.
    """

    # Parameters
    galflux = 1e4
    galradius = 0.5
    g1 = 0.1
    g2 = 0.4
    psffwhm = 0.1

    # Generates the galaxy image
    image = GenGalIm((galflux, galradius, g1, g2, psffwhm))

    # Plots
    plt.imshow(image.array)
    plt.show()


def rescale(img):
    """
    Rescale an image between 0 and 1 to plot.
    """
    return (img - np.min(img)) / np.max(img - np.min(img))


def Plot_gals():
    # file_name = '../Data/lhc_256_5.txt'
    # gals = GenSetGal(file_name)

    params = np.zeros((100, 5))
    params[:, 0] = 1e4
    params[:, 1] = 0.5
    params[:, 2] = np.linspace(-0.5, 0.5, 100)
    params[:, 3] = np.linspace(-0.5, 0.5, 100)
    params[:, 4] = 0.2

    gals = GenSetGal(params)
    gals_plots = rescale(gals[0])
    for i in range(9):
        gals_plots = np.concatenate((gals_plots, rescale(gals[i+1])), axis=1)

    for i in range(9):
        gals_plots_ = rescale(gals[10*(i+1)+1])
        for j in range(9):
            gals_plots_ = np.concatenate((gals_plots_, rescale(gals[10*(i+1)+1+j])), axis=1)
        gals_plots = np.concatenate((gals_plots, gals_plots_), axis=0)

    plt.imshow(gals_plots, cmap='gray')
    plt.axis('off')
    plt.show()


# file_name = '../Data/lhc_64_5_testing.txt'
# SaveGal(GenSetGal(file_name), 'test_64_5_testing', 'galaxies')
# PlotGal()
