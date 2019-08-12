import sys
import numpy as np
import matplotlib.pyplot as plt
import galsim
import h5py
from keras import backend as K
from keras.models import load_model
from matplotlib.colors import LogNorm
from matplotlib import cm
import umap

import matplotlib
matplotlib.use('Agg')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ------------------------- Computing assessments ---------------------------


def sersic_bulge(y):
    sersic = np.where(y[:, 0] > 0)[0]
    bulge = np.where(y[:, 5] > 0)[0]

    y_sersic = y[sersic, :4]
    y_bulge = y[bulge, 4:]
    return y_sersic, y_bulge, sersic, bulge


def rescale(img):
    """
    Rescale an image between 0 and 1 to plot.
    """
    return (img - np.min(img)) / np.max(img - np.min(img))


def mse_r2(PlotDir, true, predicted):
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
        h = plt.hist(np.sort(mse)[:-2], bins=50, density=True)
        plt.xlabel('MSE')
        plt.ylabel('Density')
        plt.title('MSE distribution (min = '+str(np.min(mse))+', max = '+str(np.max(mse))+', median = '+str(np.median(mse))+').')
        plt.tight_layout()
        plt.savefig(PlotDir+'cosmos_mse_distribution.png')
        plt.close()

        plt.figure('R$^2$ distribution')
        h = plt.hist(r2, bins=50, density=True)
        plt.xlabel('R$^2$')
        plt.ylabel('Density')
        plt.title('R$^2$ distribution (min = '+str(np.min(r2))+', max = '+str(np.max(r2))+', median = '+str(np.median(r2))+').')
        # plt.show()
        plt.tight_layout()
        plt.savefig(PlotDir+'cosmos_r2_distribution.png')
        plt.close()

    return mse, r2


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
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), 'r')
    plt.xlabel('True pixel intensity')
    plt.ylabel('Predicted pixel intensity')
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_pixels_intensity.png')
    plt.close()


def shear_estimation(PlotDir, true, predicted, psf):
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
    plt.savefig(PlotDir+'cosmos_shear_estimation.png')
    plt.close()

    diff_g1 = abs(shear_true[:, 0] - shear_pred[:, 0]) * abs(1+shear_true[:, 0]) ** -1
    diff_g2 = abs(shear_true[:, 1] - shear_pred[:, 1]) * abs(1+shear_true[:, 1]) ** -1

    return diff_g1, diff_g2


def plot_results(PlotDir, x_train, x_train_decoded, x_train_decoded_psf, x_test, x_test_decoded, x_test_decoded_psf):
    x_train_plt = rescale(x_train[0])
    x_train_decoded_plt = rescale(x_train_decoded[0])
    x_train_decoded_psf_plt = rescale(x_train_decoded_psf[0])
    error_train_plt = rescale(abs(x_train[0] - x_train_decoded_psf[0]))

    x_test_plt = rescale(x_test[0])
    x_test_decoded_plt = rescale(x_test_decoded[0])
    x_test_decoded_psf_plt = rescale(x_test_decoded_psf[0])
    error_test_plt = rescale(abs(x_test[0] - x_test_decoded_psf[0]))

    for i in range(10):
        x_train_plt = np.concatenate((x_train_plt, rescale(x_train[i+1])), axis=1)
        x_train_decoded_plt = np.concatenate((x_train_decoded_plt, rescale(x_train_decoded[i+1])), axis=1)
        x_train_decoded_psf_plt = np.concatenate((x_train_decoded_psf_plt, rescale(x_train_decoded_psf[i+1])), axis=1)
        error_train_plt = np.concatenate((error_train_plt, rescale(abs(x_train[i+1] - x_train_decoded_psf[i+1]))), axis=1)

        x_test_plt = np.concatenate((x_test_plt, rescale(x_test[i+1])), axis=1)
        x_test_decoded_plt = np.concatenate((x_test_decoded_plt, rescale(x_test_decoded[i+1])), axis=1)
        x_test_decoded_psf_plt = np.concatenate((x_test_decoded_psf_plt, rescale(x_test_decoded_psf[i+1])), axis=1)
        error_test_plt = np.concatenate((error_test_plt, rescale(abs(x_test[i+1] - x_test_decoded_psf[i+1]))), axis=1)

    plt.figure()
    plt.subplot(411)
    plt.imshow(x_train_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(412)
    plt.imshow(x_train_decoded_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(413)
    plt.imshow(x_train_decoded_psf_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(414)
    plt.imshow(error_train_plt, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_training_set_images.png')
    plt.close()

    plt.figure()
    plt.subplot(411)
    plt.imshow(x_test_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(412)
    plt.imshow(x_test_decoded_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(413)
    plt.imshow(x_test_decoded_psf_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(414)
    plt.imshow(error_test_plt, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_testing_set_images.png')
    plt.close()


def latent_space(PlotDir, x_train_encoded, x_test_encoded, y_train, y_train_sersic, y_train_bulge, y_test, y_test_sersic, y_test_bulge):
    latent_dim = x_train_encoded.shape[1]
    n_train, params_dim = y_train.shape
    n_test = y_test.shape[0]
    PlotScatter = True
    if PlotScatter:
        for w1 in range(4):
            for w2 in range(4):
                # display a 2D plot of latent space (just 2 dimensions)
                if w1 == w2:
                    w2 += 1
                i = np.random.randint(0, high=4)
                npar = np.copy(i)
                # if i < 4:
                #     y_train_ = y_train_sersic
                #     y_test_ = y_test_sersic
                #     x_train_encoded_ = x_train_encoded[train_sersic]
                #     x_test_encoded_ = x_test_encoded[test_sersic]
                # else:
                #     y_train_ = y_train_bulge
                #     y_test_ = y_test_bulge
                #     x_train_encoded_ = x_train_encoded[train_bulge]
                #     x_test_encoded_ = x_test_encoded[test_bulge]

                #     i -= 4
                y_train_ = np.zeros(n_train)
                y_test_ = np.zeros(n_test)
                for l in range(n_train):
                    if not(y_train[l, i]):
                        y_train_[l] = y_train[l, i+4]
                    else:
                        y_train_[l] = y_train[l, i]
                for l in range(n_test):
                    if not(y_test[l, i]):
                        y_test_[l] = y_test[l, i+4]
                    else:
                        y_test_[l] = y_test[l, i]

                plt.figure(figsize=(6, 6))
                # plt.scatter(x_train_encoded_[:, w1], x_train_encoded_[:, w2], c=y_train_[:, i]+1e-6, cmap='bone', norm=LogNorm(1e-6, y_train_[:, i].max()))
                plt.scatter(x_train_encoded[:, w1], x_train_encoded[:, w2], c=y_train_.flatten(), cmap='bone', norm=LogNorm())
                plt.colorbar()
                # plt.scatter(x_test_encoded_[:, w1], x_test_encoded_[:, w2], c=y_test_[:, i]+1e-6, cmap='Wistia', norm=LogNorm(1e-6, y_train_[:, i].max()))
                plt.scatter(x_test_encoded[:, w1], x_test_encoded[:, w2], c=y_test_.flatten(), cmap='Wistia', norm=LogNorm())
                plt.colorbar()
                plt.savefig(PlotDir+'cosmos_cnn_vae_scatter_latent_'+str(w1)+'_vs_'+str(w2)+'_param_'+str(npar)+'.png')
                # plt.show()
                plt.close()

    PlotFull = False
    if PlotFull:
        f, a = plt.subplots(4, 4, sharex=True, sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.rcParams.update({'font.size': 4})
        for i in range(4):
            for j in range(i+1):
                if(i != j):
                    # a[i, j].scatter(y_train[:, i], y_train[:, j], s=1, alpha=0.7)
                    # a[i, j].scatter(y_test[:, i], y_test[:, j], s=1, alpha=0.7)
                    a[i, j].grid(True)
                    a[j, i].set_visible(False)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    # a[i, i].text(0.4, 0.4, params[i], size='x-large')
                    # hist, bin_edges = np.histogram(y_train[:, i], density=True, bins=12)
                    # a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])
        plt.tight_layout()
        plt.savefig(PlotDir+'params_sersic_distrib_corr.png', figsize=(20000, 20000), bbox_inches="tight")
        plt.close()

        f, a = plt.subplots(8, 8, sharex=True, sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.rcParams.update({'font.size': 4})
        for i in range(8):
            for j in range(i+1):
                if(i != j):
                    # a[i, j].scatter(y_train[:, i+4], y_train[:, j+4], s=1, alpha=0.7)
                    # a[i, j].scatter(y_test[:, i+4], y_test[:, j+4], s=1, alpha=0.7)
                    a[i, j].grid(True)
                    a[j, i].set_visible(False)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    # a[i, i].text(0.4, 0.4, params[i], size='x-large')
                    # hist, bin_edges = np.histogram(y_train[:, i+4], density=True, bins=12)
                    # a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])
        plt.tight_layout()
        plt.savefig(PlotDir+'params_bulge+disk_distrib_corr.png', figsize=(20000, 20000), bbox_inches="tight")
        plt.close()

        f, a = plt.subplots(latent_dim, params_dim, sharex=True, sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.rcParams.update({'font.size': 4})
        for i in range(latent_dim):
            for j in range(params_dim):
                # a[i, j].scatter(x_train_encoded[:, i], y_train[:, j], s=1, alpha=0.7)
                # a[i, j].scatter(x_test_encoded[:, i], y_test[:, j], s=1, alpha=0.7)
                a[i, j].grid(True)
                # a[j, i].set_visible(False)
                plt.yticks([])
        plt.tight_layout()
        plt.savefig(PlotDir+'latent_vs_params_distrib_corr.png', figsize=(20000, 20000), bbox_inches="tight")
        plt.close()

        f, a = plt.subplots(latent_dim, latent_dim, sharex=True, sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.rcParams.update({'font.size': 4})
        for i in range(latent_dim):
            for j in range(i+1):
                if(i != j):
                    a[i, j].scatter(x_train_encoded[:, i], x_train_encoded[:, j], s=1, alpha=0.7)
                    a[i, j].scatter(x_test_encoded[:, i], x_test_encoded[:, j], s=1, alpha=0.7)
                    a[i, j].grid(True)
                    a[j, i].set_visible(False)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    # a[i+params_dim, i+params_dim].text(0.4, 0.4, 'Latent space '+str(i), size='x-large')
                    hist, bin_edges = np.histogram(x_train_encoded[:, i], density=True, bins=12)
                    a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])
        plt.tight_layout()
        plt.savefig(PlotDir+'latent_vs_latent_corr.png', figsize=(20000, 20000), bbox_inches="tight")
        plt.close()


def plot_umap(PlotDir, x_train_encoded, x_test_encoded, y_train, y_test):
    reducer = umap.UMAP()
    embedding_train = reducer.fit_transform(x_train_encoded)
    embedding_test = reducer.transform(x_test_encoded)
    plt.figure()
    plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=y_train[:, 0].flatten(), cmap='bone', norm=LogNorm())
    plt.colorbar()
    plt.scatter(embedding_test[:, 0], embedding_test[:, 1], c=y_test[:, 0].flatten(), cmap='Wistia', norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_umap_flux.png', figsize=(20000, 20000), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=y_train[:, 1].flatten(), cmap='bone', norm=LogNorm())
    plt.colorbar()
    plt.scatter(embedding_test[:, 0], embedding_test[:, 1], c=y_test[:, 1].flatten(), cmap='Wistia', norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_umap_hlr.png', figsize=(20000, 20000), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=y_train[:, 2].flatten(), cmap='bone', norm=LogNorm())
    plt.colorbar()
    plt.scatter(embedding_test[:, 0], embedding_test[:, 1], c=y_test[:, 2].flatten(), cmap='Wistia', norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_umap_q.png', figsize=(20000, 20000), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(embedding_train[:, 0], embedding_train[:, 1], c=y_train[:, 3].flatten(), cmap='bone', norm=LogNorm())
    plt.colorbar()
    plt.scatter(embedding_test[:, 0], embedding_test[:, 1], c=y_test[:, 3].flatten(), cmap='Wistia', norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_umap_phi.png', figsize=(20000, 20000), bbox_inches="tight")
    plt.close()


def plot_gps(PlotDir, x_test, x_test_encoded, x_test_gp_encoded, x_test_decoded_psf, x_test_gp_decoded):
    x_test_plt = rescale(x_test[0])
    x_test_decoded_plt = rescale(x_test_decoded_psf[0])
    x_test_decoded_gp_plt = rescale(x_test_gp_decoded[0])

    for i in range(10):
        x_test_plt = np.concatenate((x_test_plt, rescale(x_test[i+1])), axis=1)
        x_test_decoded_plt = np.concatenate((x_test_decoded_plt, rescale(x_test_decoded_psf[i+1])), axis=1)
        x_test_decoded_gp_plt = np.concatenate((x_test_decoded_gp_plt, rescale(x_test_gp_decoded[i+1])), axis=1)

    plt.figure()
    plt.subplot(311)
    plt.imshow(x_test_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(312)
    plt.imshow(x_test_decoded_plt, cmap='gray')
    plt.axis('off')
    plt.subplot(313)
    plt.imshow(x_test_decoded_gp_plt, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_testing_set_images_gp.png')
    plt.close()

    counts_true, bins = np.histogram(x_test_encoded.flatten(), bins=100)
    counts_pred, bins = np.histogram(x_test_gp_encoded.flatten(), bins=bins)

    plt.figure()
    plt.subplot(121)
    plt.semilogy(bins[:-1], counts_true, 's-', label='VAE encoded')
    plt.semilogy(bins[:-1], counts_pred, 's-', label='GP encoded')
    plt.legend()
    plt.xlabel('Latent space variable value')
    plt.ylabel('Counts')
    # plt.show()
    plt.subplot(122)
    plt.scatter(x_test_encoded.flatten(), x_test_gp_encoded.flatten(), s=1)
    plt.plot(np.linspace(np.min(x_test_encoded.flatten()), np.max(x_test_encoded.flatten())), np.linspace(np.min(x_test_encoded.flatten()), np.max(x_test_encoded.flatten())), 'r')
    plt.xlabel('VAE latent space variable value')
    plt.ylabel('GP latent space variable value')
    plt.tight_layout()
    plt.savefig(PlotDir+'cosmos_gp_encoded_vae_encoded_testing_set.png')
    plt.close()


# ------------------------------------- MAIN -----------------------------------------


def main(args):

    # ------------------------ Parameters ---------------------------------------
    DataDir = '../Data/Cosmos/'
    PlotDir = '../Plots/Cosmos_plots/'
    ntrain = 4096
    ntest = 256
    nx = 64
    ny = 64

    os.remove(PlotDir+'*.png')

    # ------------------------ Load data and models -----------------------------

    # Load training and testing set
    f = h5py.File(DataDir + 'data/cosmos_real_trainingset_train_'+str(ntrain)+'_test_'+str(ntest)+'.hdf5', 'r')
    x_train = np.array(f['real galaxies'])
    y_train = np.array(f['parameters'])
    f.close()

    f = h5py.File(DataDir + 'data/cosmos_real_testingset_train_'+str(ntrain)+'_test_'+str(ntest)+'.hdf5', 'r')
    x_test = np.array(f['real galaxies'])
    y_test = np.array(f['parameters'])
    f.close()

    # Sercic - Bulge+disk
    y_train_sersic, y_train_bulge, train_sersic, train_bulge = sersic_bulge(y_train)
    y_test_sersic, y_test_bulge, test_sersic, test_bulge = sersic_bulge(y_test)

    # Rescaling
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_test = (x_test - xmin) / xmax
    x_train = K.cast_to_floatx(x_train)
    x_test = K.cast_to_floatx(x_test)

    ymin = np.min(y_train, axis=0)
    ymax = np.max(y_train - ymin, axis=0)
    y_train = (y_train - ymin) / ymax
    y_train_sersic = (y_train_sersic - ymin[:4]) / ymax[:4]
    y_train_bulge = (y_train_bulge - ymin[4:]) / ymax[4:]
    y_test = (y_test - ymin) / ymax
    y_test_sersic = (y_test_sersic - ymin[:4]) / ymax[:4]
    y_test_bulge = (y_test_bulge - ymin[4:]) / ymax[4:]

    # Compute reconstructed testing set
    # x_test_encoded = np.loadtxt(DataDir+'Cosmos/models/cvae_encoded_xtest_512_5.txt')
    # x_test_encoded = K.cast_to_floatx(x_test_encoded)
    # x_test_encoded = x_test_encoded.astype('float32')

    # decoder = load_model(DataDir+'Galsim/cvae_decoder_model_galsim.h5')
    # x_test_decoded = np.zeros((ntest, nx, ny))
    # x_test_decoded = decoder.predict(x_test_encoded)
    x_test_decoded = np.reshape(np.loadtxt(DataDir+'models/cvae_cosmos_decoded_xtest_'+str(ntest)+'.txt'), (ntest, nx, ny))
    x_test_decoded_psf = np.reshape(np.loadtxt(DataDir+'models/cvae_cosmos_decoded_psf_xtest_'+str(ntest)+'.txt'), (ntest, nx, ny))
    x_test_encoded = np.loadtxt(DataDir+'models/cvae_cosmos_encoded_xtest_'+str(ntest)+'.txt')
    x_test_gp_encoded = np.loadtxt(DataDir + 'models/cvae_cosmos_gp_encoded_xtest_'+str(ntrain)+'_'+str(ntest)+'.txt')
    x_test_gp_decoded = np.reshape(np.loadtxt(DataDir + 'models/cvae_cosmos_gp_decoded_xtest_'+str(ntrain)+'_'+str(ntest)+'.txt'), (ntest, nx, ny))

    # Load reconstructed training set
    x_train_decoded = np.reshape(np.loadtxt(DataDir+'models/cvae_cosmos_decoded_xtrain_'+str(ntrain)+'.txt'), (ntrain, nx, ny))
    x_train_decoded_psf = np.reshape(np.loadtxt(DataDir+'models/cvae_cosmos_decoded_psf_xtrain_'+str(ntrain)+'.txt'), (ntrain, nx, ny))
    x_train_encoded = np.loadtxt(DataDir+'models/cvae_cosmos_encoded_xtrain_'+str(ntrain)+'.txt')

    # -------------------- Plotting routines --------------------------

    print('Plot results ...')
    plot_results(PlotDir, x_train, x_train_decoded, x_train_decoded_psf, x_test, x_test_decoded, x_test_decoded_psf)
    print('Plot mse/r2 ...')
    mse, r2 = mse_r2(PlotDir, x_train, x_train_decoded_psf)
    print('Plot pixel intensity ...')
    pixel_intensity(PlotDir, x_test, x_test_decoded_psf)
    print('Plot shear estimation ...')
    diff_g1, diff_g2 = shear_estimation(PlotDir, x_train, x_train_decoded[:, :, :], np.zeros(x_test.shape))
    # print('Plot latent space ...')
    # latent_space(PlotDir, x_train_encoded, x_test_encoded, y_train, y_train_sersic, y_train_bulge, y_test, y_test_sersic, y_test_bulge)
    print('Plot umap ...')
    plot_umap(PlotDir, x_train_encoded, x_test_encoded, y_train, y_test)
    print('Plot GP ...')
    plot_gps(PlotDir, x_test, x_test_encoded, x_test_gp_encoded, x_test_decoded_psf, x_test_gp_decoded)


if __name__ == "__main__":
    image = main(sys.argv)
