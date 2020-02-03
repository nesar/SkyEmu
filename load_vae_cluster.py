import sys
import numpy as np
import matplotlib.pyplot as plt
import galsim
import h5py
from matplotlib.colors import LogNorm
from matplotlib import cm
import umap
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import load_model, Model
import tensorflow as tf

# import matplotlib
# matplotlib.use('Agg')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def rescale_im(img):
    """
    Rescale an image between 0 and 1 to plot.
    """
    return (img - np.min(img)) / np.max(img - np.min(img))


# Convolution with PSF function
def psf_convolve(args):
    img, psf = args
    imgfft = tf.spectral.rfft2d(img[:, :, :, 0])
    psffft = tf.spectral.rfft2d(psf[:, :, :, 0])
    convfft = tf.spectral.irfft2d(imgfft * psffft)
    h = tf.expand_dims(convfft, axis=-1)
    return h


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
     Arguments
        args (tensor): mean and log of variance of Q(z|X)
     Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim), mean=epsilon_mean, stddev=epsilon_std)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# ------------------------ Parameters ---------------------------------------
DataDir = '../Data/Cosmos/'
PlotDir = '../Plots/Cosmos_plots/'
ntrain = 16384
ntest = 512
nx = 64
ny = 64
input_shape = (nx, ny, 1)
epsilon_mean = 0
epsilon_std = 1e-5

# os.remove(PlotDir+'*.png')

# ------------------------ Load data and models -----------------------------

print('Loading data ...')
# Load training and testing set
f = h5py.File(DataDir + 'data/cosmos_real_trainingset_train_'+str(ntrain)+'_test_'+str(ntest)+'.hdf5', 'r')
x_train = np.array(f['real galaxies'])
y_train = np.array(f['parameters'])
psf_train = np.fft.fftshift(np.array(f['psf']))
f.close()

f = h5py.File(DataDir + 'data/cosmos_real_testingset_train_'+str(ntrain)+'_test_'+str(ntest)+'.hdf5', 'r')
x_test = np.array(f['real galaxies'])
y_test = np.array(f['parameters'])
psf_test = np.fft.fftshift(np.array(f['psf']))
f.close()

# Rescaling

xmin = np.min(x_train)
xmax = np.max(x_train) - xmin
x_train = (x_train - xmin) / xmax
x_test = (x_test - xmin) / xmax

ymin = np.min(y_train, axis=0)
ymax = np.max(y_train - ymin, axis=0)
y_train = (y_train - ymin) / ymax
y_test = (y_test - ymin) / ymax

x_train = K.cast_to_floatx(x_train)
x_test = K.cast_to_floatx(x_test)
psf_train = K.cast_to_floatx(psf_train)
psf_test = K.cast_to_floatx(psf_test)

x_train = np.reshape(x_train, [-1, nx, ny, 1])
x_test = np.reshape(x_test, [-1, nx, ny, 1])
psf_train = np.reshape(psf_train, [-1, nx, ny, 1])
psf_test = np.reshape(psf_test, [-1, nx, ny, 1])

x_train = x_train.astype('float32')  # / 255
x_test = x_test.astype('float32')  # / 255
psf_train = psf_train.astype('float32')
psf_test = psf_test.astype('float32')

x_train_encoded = np.loadtxt(DataDir + 'models/cvae_cosmos_encoded_xtest_'+str(ntrain)+'.txt')
x_test_encoded = np.loadtxt(DataDir + 'models/cvae_cosmos_encoded_xtest_'+str(ntest)+'.txt')

print('Loading models ...')
# encoder = load_model(DataDir+'models/cvae_encoder_model_cosmos_'+str(ntrain)+'_train_'+str(ntest)+'_test.h5')
decoder1 = load_model(DataDir+'models/cvae_decoder_model_cosmos_'+str(ntrain)+'_train_'+str(ntest)+'_test.h5')

dec_inputs2 = Input(shape=input_shape, name='dec_inputs2')
psf_inputs = Input(shape=input_shape, name='psf_inputs')
outputs2 = Lambda(psf_convolve, output_shape=input_shape)([dec_inputs2, psf_inputs])
decoder2 = Model([dec_inputs2, psf_inputs], outputs2)

print('Compute decoded images ...')
x_test_decoded = decoder2.predict([decoder1.predict(x_test_encoded), psf_test])[:, :, :, 0]
np.savetxt(DataDir+'models/cvae_cosmos_decoded_psf_xtest_'+str(ntest)+'.txt', np.reshape(x_test_decoded, (x_test_decoded.shape[0], nx*ny)))
x_train_decoded = decoder2.predict([decoder1.predict(x_train_encoded), psf_train])[:, :, :, 0]
np.savetxt(DataDir+'models/cvae_cosmos_decoded_psf_xtrain_'+str(ntrain)+'.txt', np.reshape(x_train_decoded, (x_train_decoded.shape[0], nx*ny)))
del x_train_decoded
x_test_decoded_noise = x_test_decoded + 1e-4*np.random.randn(x_test_decoded.shape[0], x_test_decoded.shape[1], x_test_decoded.shape[2])

x_test = x_test[:, :, :, 0]
plt.figure()
orig_plots = rescale_im(x_test[0])
for i in range(9):
    orig_plots = np.concatenate((orig_plots, rescale_im(x_test[i+1])), axis=1)
for i in range(1):
    orig_plots_ = rescale_im(x_test[10*(i+1)+1])
    for j in range(9):
        orig_plots_ = np.concatenate((orig_plots_, rescale_im(x_test[10*(i+1)+1+j])), axis=1)
    orig_plots = np.concatenate((orig_plots, orig_plots_), axis=0)
plt.imshow(orig_plots)
plt.axis('off')
plt.show()

plt.figure()
rec_plots = rescale_im(x_test_decoded[0])
for i in range(9):
    rec_plots = np.concatenate((rec_plots, rescale_im(x_test_decoded[i+1])), axis=1)
for i in range(1):
    rec_plots_ = rescale_im(x_test_decoded[10*(i+1)+1])
    for j in range(9):
        rec_plots_ = np.concatenate((rec_plots_, rescale_im(x_test_decoded[10*(i+1)+1+j])), axis=1)
    rec_plots = np.concatenate((rec_plots, rec_plots_), axis=0)
plt.imshow(rec_plots)
plt.axis('off')
plt.show()

plt.figure()
rec_plots = rescale_im(x_test_decoded_noise[0])
for i in range(9):
    rec_plots = np.concatenate((rec_plots, rescale_im(x_test_decoded_noise[i+1])), axis=1)
for i in range(1):
    rec_plots_ = rescale_im(x_test_decoded_noise[10*(i+1)+1])
    for j in range(9):
        rec_plots_ = np.concatenate((rec_plots_, rescale_im(x_test_decoded_noise[10*(i+1)+1+j])), axis=1)
    rec_plots = np.concatenate((rec_plots, rec_plots_), axis=0)
plt.imshow(rec_plots)
plt.axis('off')
plt.show()
