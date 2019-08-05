import numpy as np
import matplotlib.pyplot as plt
import galsim
import h5py
from keras import backend as K
from keras.models import load_model

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ------------------------- Computing assessments ---------------------------


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

    plot = True
    if plot:
        plt.figure('MSE distribution')
        h = plt.hist(mse, bins=50, density=True)
        plt.xlabel('MSE')
        plt.ylabel('Density')
        plt.title('MSE distribution (min = '+str(np.min(mse))+', max = '+str(np.max(mse))+', median = '+str(np.median(mse))+').')

        plt.figure('R$^2$ distribution')
        h = plt.hist(r2, bins=50, density=True)
        plt.xlabel('R$^2$')
        plt.ylabel('Density')
        plt.title('R$^2$ distribution (min = '+str(np.min(r2))+', max = '+str(np.max(r2))+', median = '+str(np.median(r2))+').')
        plt.show()

    return mse, r2


def pixel_intensity(true, predicted):
    """
    Plots the distribution of pixel intensities for the validation set (or any set made of real/input image set) and the reconstructed/predicted data set.
    """
    counts_true, bins = np.histogram(true.flatten(), bins=100)
    counts_pred, bins = np.histogram(predicted.flatten(), bins=bins)

    plt.figure('Pixel flux value distribution')
    plt.plot(bins[:-1], counts_true, 's-', label='True')
    plt.plot(bins[:-1], counts_pred, 's-', label='Predicted')
    plt.legend()
    plt.xlabel('Pixel intensity')
    plt.ylabel('Counts')
    plt.show()


def shear_estimation(true, predicted, psf):
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

        psf = galsim.Gaussian(fwhm=0.04)

        shape_true = galsim.hsm.EstimateShear(galsim.Image(true[i]), psf.drawImage(nx=32, ny=32, scale=0.04), hsmparams=big_run_params).observed_shape
        shape_pred = galsim.hsm.EstimateShear(galsim.Image(predicted[i]), psf.drawImage(nx=32, ny=32, scale=0.04), hsmparams=big_run_params).observed_shape

        shear_true[i] = np.array([shape_true.g1, shape_true.g2])
        shear_pred[i] = np.array([shape_pred.g1, shape_pred.g2])

    countsg1_true, bins1 = np.histogram(shear_true[:, 0], bins=20)
    countsg1_pred, bins1 = np.histogram(shear_pred[:, 0], bins=bins1)
    countsg2_true, bins2 = np.histogram(shear_true[:, 1], bins=20)
    countsg2_pred, bins2 = np.histogram(shear_pred[:, 1], bins=bins2)
    counts_true, bins3 = np.histogram(shear_true[:, 0]**2 + shear_true[:, 1]**2, bins=20)
    counts_pred, bins3 = np.histogram(shear_pred[:, 0]**2 + shear_pred[:, 1]**2, bins=bins3)

    plt.figure('Shear estimation')
    plt.subplot(221)
    plt.plot(bins1[:-1], countsg1_true, 's--', label='True g1')
    plt.plot(bins1[:-1], countsg1_pred, 's--', label='Predicted g1')
    plt.legend()
    plt.xlabel('g1 value')
    plt.ylabel('Counts')
    plt.subplot(222)
    plt.plot(bins2[:-1], countsg2_true, 's--', label='True g2')
    plt.plot(bins2[:-1], countsg2_pred, 's--', label='Predicted g2')
    plt.legend()
    plt.xlabel('g2 value')
    plt.ylabel('Counts')
    plt.subplot(223)
    plt.plot(bins3[:-1], counts_true, 's--', label='True g1$^2$ + g2$^2$')
    plt.plot(bins3[:-1], counts_pred, 's--', label='Predicted g1$^2$ + g2$^2$')
    plt.legend()
    plt.xlabel('g1$^2$ + g2$^2$')
    plt.ylabel('Counts')
    plt.subplot(224)
    diff_module = abs((shear_true[:, 0]**2 + shear_true[:, 1]**2) - (shear_pred[:, 0]**2 + shear_pred[:, 1]**2)) * ((shear_true[:, 0]**2 + shear_true[:, 1]**2) **(-1)) * 100
    diff_g1 = abs(shear_true[:, 0] - shear_pred[:, 0]) * 100 * shear_true[:, 0] ** -1
    diff_g2 = abs(shear_true[:, 1] - shear_pred[:, 1]) * 100 * shear_true[:, 1] ** -1
    plt.plot(np.sort(diff_module), 's--', label='g1$^2$ + g2$^2$')
    plt.plot(np.sort(diff_g1), 's--', label='g1')
    plt.plot(np.sort(diff_g2), 's--', label='g2')
    plt.legend()
    # plt.xlabel('g1$^2$ + g2$^2$')
    plt.ylabel('Error ($\%$)')
    plt.show()

    return shear_true, shear_pred

# ------------------------------------- MAIN -----------------------------------------


def main():

    # ------------------------ Parameters ---------------------------------------
    DataDir = '../Data/'
    ntrain = 512
    ntest = 64
    nx = 32
    ny = 32

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
    x_test_encoded = np.loadtxt(DataDir+'Galsim/cvae_encoded_xtest_512_5.txt')
    x_test_encoded = K.cast_to_floatx(x_test_encoded)
    x_test_encoded = x_test_encoded.astype('float32')

    decoder = load_model(DataDir+'Galsim/cvae_decoder_model_galsim.h5')
    # x_test_decoded = np.zeros((ntest, nx, ny))
    x_test_decoded = decoder.predict(x_test_encoded)

    # Load reconstructed training set
    x_train_decoded = np.reshape(np.loadtxt(DataDir+'GalSim/cvae_decoded_xtrain_512_5.txt'), (ntrain, nx, ny))

    mse, r2 = mse_r2(x_train, x_train_decoded)
    pixel_intensity(x_train, x_train_decoded)
    shear_true, shear_pred = shear_estimation(x_train, x_train_decoded[:, :, :], np.zeros(x_test.shape))
    return shear_true, shear_pred
