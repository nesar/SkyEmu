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


def GenGalIm(galflux, galradius, g1, g2, psffwhm):
    """
    Input parameters :

    - flux : Flux of the galaxy in counts.
    - radius : Radius of the galaxy in arcsec.
    - g1 : Reduced shear component.
    - g2 : Reduced shear component. See demo2 comments. sqrt(g1+g2*1.j) < 1.0.
    - psf_fwhm : fwhm of the Gaussian PSF used for convolution.

    Output :
    - image : 33x33 pixels image of a galaxy given input parameters.
    """

    # Fixed parameters
    nx = 33 # pixels in the 1st spatial dimension
    ny = 33 # pixels in the 2nd spatial dimension
    pixel_scale = 0.2 # arcsec/pixel

    # Define the galaxy profile
    gal = galsim.Exponential(flux=galflux, scale_radius=galradius)
    gal = gal.shear(g1=g1, g2=g2)

    # Define the PSF profile
    psf = galsim.Gaussian(fwhm=psffwhm)

    # Convolution
    final = galsim.Convolve([gal, psf])

    # Draw the image
    image = final.drawImage(nx=nx, ny=ny, scale=pixel_scale)

    return image


def SaveGal(image, fname, dname):
    """
    Save a single galaxy image into a hdf5 file.
    Input parameters :

    - image : Galaxy image to save
    - fname : file name
    - dname : dataset name
    """
    if not os.path.isdir('output_tests'):
        os.mkdir('output_tests')
    file_name = os.path.join('output_tests', fname)
    f = h5py.File(file_name, 'w')
    f.create_dataset(dname, data=image)
    f.close()


def PlotGal():
    """
    Given parameters (flux, radius, psf fwhm, shear profile), plots the related galaxy image.
    """

    # Parameters
    galflux = 1e5
    galradius = 1
    g1 = 0.1
    g2 = 0.2
    psffwhm = 0.3

    # Generates the galaxy image
    image = GenGalIm(galflux, galradius, g1, g2, psffwhm)

    # Plots
    plt.imshow(image.array)
    plt.show()
