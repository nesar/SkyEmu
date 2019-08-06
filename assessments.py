import numpy as np
import matplotlib.pyplot as plt
import galsim
import h5py
from keras import backend as K
from keras.models import load_model

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ------------------------- Computing assessments ---------------------------


def rescale(img):
    """
    Rescale an image between 0 and 1 to plot.
    """
    return (img - np.min(img)) / np.max(img - np.min(img))


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
        plt.savefig('../Plots/Cosmos_plots/cosmos_mse_distribution.png')

        plt.figure('R$^2$ distribution')
        h = plt.hist(r2, bins=50, density=True)
        plt.xlabel('R$^2$')
        plt.ylabel('Density')
        plt.title('R$^2$ distribution (min = '+str(np.min(r2))+', max = '+str(np.max(r2))+', median = '+str(np.median(r2))+').')
        # plt.show()
        plt.savefig('../Plots/Cosmos_plots/cosmos_r2_distribution.png')
        plt.close()

    return mse, r2


def pixel_intensity(true, predicted):
    """
    Plots the distribution of pixel intensities for the validation set (or any set made of real/input image set) and the reconstructed/predicted data set.
    """
    counts_true, bins = np.histogram(true.flatten(), bins=100)
    counts_pred, bins = np.histogram(predicted.flatten(), bins=bins)

    plt.figure('Pixel flux value distribution')
    plt.semilogy(bins[:-1], counts_true, 's-', label='True')
    plt.semilogy(bins[:-1], counts_pred, 's-', label='Predicted')
    plt.legend()
    plt.xlabel('Pixel intensity')
    plt.ylabel('Counts')
    # plt.show()
    plt.savefig('../Plots/Cosmos_plots/cosmos_pixels_intensity.png')
    plt.close()


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
        # print(i)
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
    plt.scatter(shear_true[:, 0], shear_pred[:, 0])
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
    plt.scatter(shear_true[:, 1], shear_pred[:, 1])
    plt.xlabel('True g2')
    plt.ylabel('Predicted g2')
    # plt.legend()
    plt.savefig('../Plots/Cosmos_plots/cosmos_shear_estimation.png')
    plt.close()

    diff_g1 = abs(shear_true[:, 0] - shear_pred[:, 0]) * abs(1+shear_true[:, 0]) ** -1
    diff_g2 = abs(shear_true[:, 1] - shear_pred[:, 1]) * abs(1+shear_true[:, 1]) ** -1

    return diff_g1, diff_g2


def plot_results(x_train, x_train_decoded, x_test, x_test_decoded):
    x_train_plt = rescale(x_train[0])
    x_train_decoded_plt = rescale(x_train_decoded[0])
    error_train_plt = rescale(abs(x_train[0] - x_train_decoded[0]))

    x_test_plt = rescale(x_test[0])
    x_test_decoded_plt = rescale(x_test_decoded[0])
    error_test_plt = rescale(abs(x_test[0] - x_test_decoded[0]))

    for i in range(10):
        x_train_plt = np.concatenate((x_train_plt, rescale(x_train[i+1])), axis=1)
        x_train_decoded_plt = np.concatenate((x_train_decoded_plt, rescale(x_train_decoded[i+1])), axis=1)
        error_train_plt = np.concatenate((error_train_plt, rescale(abs(x_train[i+1] - x_train_decoded[i+1]))), axis=1)

        x_test_plt = np.concatenate((x_test_plt, rescale(x_test[i+1])), axis=1)
        x_test_decoded_plt = np.concatenate((x_test_decoded_plt, rescale(x_test_decoded[i+1])), axis=1)
        error_test_plt = np.concatenate((error_test_plt, rescale(abs(x_test[i+1] - x_test_decoded[i+1]))), axis=1)

    plt.figure()
    plt.subplot(311)
    plt.imshow(x_train_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(x_train_decoded_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(313)
    plt.imshow(error_train_plt, cmap='gray')
    plt.axis('off')
    plt.savefig('../Plots/Cosmos_plots/cosmos_training_set_images.png')
    plt.close()

    plt.figure()
    plt.subplot(311)
    plt.imshow(x_test_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(x_test_decoded_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(313)
    plt.imshow(error_test_plt, cmap='gray')
    plt.axis('off')
    plt.savefig('../Plots/Cosmos_plots/cosmos_testing_set_images.png')
    plt.close()


def latent_space(x_train_encoded, x_test_encoded, y_train, y_test):

    PlotScatter = True
    if PlotScatter:
        w1 = 1
        w2 = 2
        # display a 2D plot of latent space (just 2 dimensions)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_train_encoded[:, w1], x_train_encoded[:, w2], c=y_train[:, 0], cmap='spring')
        plt.colorbar()
        plt.scatter(x_test_encoded[:, w1], x_test_encoded[:, w2], c=y_test[:, 0], cmap='copper')
        plt.colorbar()
        # plt.title(fileOut)
        plt.savefig('../Plots/Cosmos_plots/cnn_vae_scatter_latent_'+str(w1)+'_vs_'+str(w2)+'.png')
        plt.close()

    PlotFull = False
    if PlotFull:
        params_dim = x_train_encoded.shape[1]
        latent_dim = y_train.shape[1]
        total_dim = params_dim + latent_dim
        f, a = plt.subplots(total_dim, total_dim, sharex=True, sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.rcParams.update({'font.size': 4})
        for i in range(params_dim):
            for j in range(i+1):
                if(i != j):
                    a[i, j].scatter(y_train[:, i], y_train[:, j], s=1, alpha=0.7)
                    a[i, j].scatter(y_test[:, i], y_test[:, j], s=1, alpha=0.7)
                    a[i, j].grid(True)
                    a[j, i].set_visible(False)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    # a[i, i].text(0.4, 0.4, params[i], size='x-large')
                    hist, bin_edges = np.histogram(y_train[:, i], density=True, bins=12)
                    a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])

        for i in range(latent_dim):
            for j in range(params_dim):
                a[i+params_dim, j].scatter(x_train_encoded[:, i], y_train[:, j], s=1, alpha=0.7)
                a[i+params_dim, j].scatter(x_test_encoded[:, i], y_test[:, j], s=1, alpha=0.7)
                a[i+params_dim, j].grid(True)
                a[j, i+params_dim].set_visible(False)
                plt.yticks([])

        for i in range(latent_dim):
            for j in range(i+1):
                if(i != j):
                    a[i+params_dim, j+params_dim].scatter(x_train_encoded[:, i], x_train_encoded[:, j], s=1, alpha=0.7)
                    a[i+params_dim, j+params_dim].scatter(x_test_encoded[:, i], x_test_encoded[:, j], s=1, alpha=0.7)
                    a[i+params_dim, j+params_dim].grid(True)
                    a[j+params_dim, i+params_dim].set_visible(False)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    # a[i+params_dim, i+params_dim].text(0.4, 0.4, 'Latent space '+str(i), size='x-large')
                    hist, bin_edges = np.histogram(x_train_encoded[:, i], density=True, bins=12)
                    a[i+params_dim, i+params_dim].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])

        plt.tight_layout()
        plt.savefig('../Plots/Cosmos_plots/Latent_vs_params_full_corr.pdf', figsize=(20000, 20000), bbox_inches="tight")
        plt.close()


# ------------------------------------- MAIN -----------------------------------------


def main():

    # ------------------------ Parameters ---------------------------------------
    DataDir = '../Data/Cosmos/'
    ntrain = 2048
    ntest = 128
    nx = 64
    ny = 64

    # ------------------------ Load data and models -----------------------------

    # Load training and testing set
    f = h5py.File(DataDir + 'data/cosmos_real_train_2048.hdf5', 'r')
    x_train = np.array(f['real galaxies'])
    y_train = np.array(f['parameters'])
    f.close()

    f = h5py.File(DataDir + 'data/cosmos_real_test_128.hdf5', 'r')
    x_test = np.array(f['real galaxies'])
    y_test = np.array(f['parameters'])
    f.close()

    # Rescaling
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_test = (x_test - xmin) / xmax
    x_train = K.cast_to_floatx(x_train)
    x_test = K.cast_to_floatx(x_test)

    ymin = np.min(y_train)
    ymax = np.max(y_train - ymin)
    y_train = (y_train - ymin) / ymax
    y_test = (y_test - ymin) / ymax

    # Compute reconstructed testing set
    # x_test_encoded = np.loadtxt(DataDir+'Cosmos/models/cvae_encoded_xtest_512_5.txt')
    # x_test_encoded = K.cast_to_floatx(x_test_encoded)
    # x_test_encoded = x_test_encoded.astype('float32')

    # decoder = load_model(DataDir+'Galsim/cvae_decoder_model_galsim.h5')
    # x_test_decoded = np.zeros((ntest, nx, ny))
    # x_test_decoded = decoder.predict(x_test_encoded)
    x_test_decoded = np.reshape(np.loadtxt(DataDir+'models/cvae_cosmos_decoded_xtest_'+str(ntest)+'_5.txt'), (ntest, nx, ny))
    x_test_encoded = np.loadtxt(DataDir+'models/cvae_cosmos_encoded_xtest_'+str(ntest)+'_5.txt')

    # Load reconstructed training set
    x_train_decoded = np.reshape(np.loadtxt(DataDir+'models/cvae_cosmos_decoded_xtrain_'+str(ntrain)+'_5.txt'), (ntrain, nx, ny))
    x_train_encoded = np.loadtxt(DataDir+'models/cvae_cosmos_encoded_xtrain_'+str(ntrain)+'_5.txt')

    # -------------------- Plotting routines --------------------------

    plot_results(x_train, x_train_decoded, x_test, x_test_decoded)
    mse, r2 = mse_r2(x_train, x_train_decoded)
    pixel_intensity(x_train, x_train_decoded)
    diff_g1, diff_g2 = shear_estimation(x_train, x_train_decoded[:, :, :], np.zeros(x_test.shape))
    latent_space(x_train_encoded, x_test_encoded, y_train, y_test)

    return diff_g1, diff_g2
