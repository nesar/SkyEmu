from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D, Flatten, Lambda, GaussianNoise
from keras.layers import Reshape, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
# import GPy

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Convolution with PSF function
def psf_convolve(args):
    img, psf = args
    imgfft = tf.spectral.rfft2d(img[:, :, :])
    psffft = tf.spectral.rfft2d(psf[:, :, :])
    convfft = tf.spectral.irfft2d(imgfft * psffft)
    h = tf.expand_dims(convfft, axis=-1)
    return h


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


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
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


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_cosmos"):
    """ Plots labels and cosmos galaxies as function of 2-dim latent vector
        Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "_params_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    # plt.show()

# ------------------------------ LOAD DATA ----------------------------------


# network parameters
DataDir = '../Data/Cosmos/'

n_train = 200
n_test = 20
nx = 64
ny = 64
nparams = 10


# Load training set images and rescale fluxes
path = DataDir + 'data/cosmos_real_trainingset_train_'+str(n_train)+'_test_'+str(n_test)+'.hdf5'
f = h5py.File(path, 'r')
x_train = np.array(f['real galaxies'])
xmin = np.min(x_train)
xmax = np.max(x_train) - xmin
x_train = (x_train - xmin) / xmax
x_train = np.reshape(x_train, (n_train, nx*ny))

# Load training set parameters and rescale
y_train = np.array(f['parameters'])
ymean = np.mean(y_train, axis=0)
ymax = np.max(abs(y_train - ymean), axis=0)
y_train = (y_train - ymean) * ymax**-1

# Load training set psfs
psf_train = np.fft.fftshift(np.array(f['psf']))
psf_ = psf_train[0]
f.close()

# Load testing set and rescale fluxes
path = DataDir + 'data/cosmos_real_testingset_train_'+str(n_train)+'_test_'+str(n_test)+'.hdf5'
f = h5py.File(path, 'r')
x_test = np.array(f['real galaxies'])
x_test = (x_test - xmin) / xmax
x_test = np.reshape(x_test, (n_test, nx*ny))

# Load testing set parameters and rescale
y_test = np.array(f['parameters'])
y_test = (y_test - ymean) * ymax**-1

# Load training set psfs
psf_test = np.fft.fftshift(np.array(f['psf']))
f.close()

# Cast to float, reshaping, ...
x_train = K.cast_to_floatx(x_train)
x_test = K.cast_to_floatx(x_test)
y_train = K.cast_to_floatx(y_train)
y_test = K.cast_to_floatx(y_test)
psf_train = K.cast_to_floatx(psf_train)
psf_test = K.cast_to_floatx(psf_test)

x_train = np.reshape(x_train, [-1, nx*ny, ])
x_test = np.reshape(x_test, [-1, nx*ny, ])
y_train = np.reshape(y_train, [-1, nparams, ])
y_test = np.reshape(y_test, [-1, nparams, ])
psf_train = np.reshape(psf_train, [-1, nx, ny, ])
psf_test = np.reshape(psf_test, [-1, nx, ny, ])

x_train = x_train.astype('float32')  # / 255
x_test = x_test.astype('float32')  # / 255
y_train = y_train.astype('float32')  # / 255
y_test = y_test.astype('float32')  # / 255
psf_train = psf_train.astype('float32')
psf_test = psf_test.astype('float32')

# network parameters
input_shape = (nx, ny, )
batch = 12
kernel_size = 4
n_conv = 2
filters = 8
shape = (60, 60, 1)
interm_dim1 = 3600
interm_dim2 = 4096
interm_dim3 = 1024
interm_dim4 = 256
interm_dim5 = 128
latent_dim = 32
epochs = 1000
drop = 0.1
l1_ = 0.01
l2_ = 0.01

epsilon_mean = 0.
epsilon_std = 1e-5

learning_rate = 1e-4
decay_rate = 1e-1

# VAE model = encoder + decoder
# build encoder model

xin = Input(shape=(nx*ny, ), name='x_input')
cond = Input(shape=(nparams, ), name='conditions')

inputs = concatenate([xin, cond])
x = Dense(interm_dim1, activation='relu')(inputs)
x = Reshape((shape[0], shape[1], shape[2]))(x)
for i in range(n_conv):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape2 = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(interm_dim2, activation='relu')(x)
x = Dropout(drop)(x)
x = Dense(interm_dim3, activation='relu')(x)
x = Dense(interm_dim4, activation='relu')(x)
x = Dense(interm_dim5, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model([xin, cond], [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file=DataDir+'Figs/vae_cnn_encoder_cosmos.png', show_shapes=True)

# DECODER

zin = Input(shape=(latent_dim, ), name='z_sampling')
# psf_inputs1 = Input(shape=input_shape, name='psf_input1')
latent_inputs1 = concatenate([zin, cond])

x1 = Dense(interm_dim5, activation='relu')(latent_inputs1)
x1 = Dense(interm_dim4, activation='relu')(x1)
x1 = Dense(interm_dim3, activation='relu')(x1)
x1 = Dropout(drop)(x1)
x1 = Dense(interm_dim2, activation='relu')(x1)
x1 = Dense(shape2[1] * shape2[2] * shape2[3], activation='relu')(x1)
x1 = Reshape((shape2[1], shape2[2], shape2[3]))(x1)

for i in range(n_conv):
    x1 = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x1)
    filters //= 2
x1 = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x1)

x1 = Flatten()(x1)

x1 = Dense(nx*ny, activation='relu')(x1)
outputs1 = Reshape(input_shape)(x1)

# outputs1 = GaussianNoise(stddev=1)(x1)
# outputs1 = Lambda(poisson_noise, output_shape=input_shape)(x1)

# instantiate decoder model
decoder1 = Model([zin, cond], outputs1, name='decoder')
decoder1.summary()

dec_inputs2 = Input(shape=input_shape, name='dec_inputs2')
psf_inputs = Input(shape=input_shape, name='psf_inputs')

outputs2 = Lambda(psf_convolve, output_shape=input_shape)([dec_inputs2, psf_inputs])

decoder2 = Model([dec_inputs2, psf_inputs], outputs2)
decoder2.summary()
# plot_model(decoder1, to_file='vae_cnn_decoder.png', show_shapes=True)
# decoder_nopsfnoise = Model(latent_inputs1, outputs, name='decoder_nopsfnoise')

# instantiate VAE model
outputs = decoder2([decoder1([encoder([xin, cond])[2], cond]), psf_inputs])
vae = Model([xin, cond, psf_inputs], outputs, name='vae')

####### main function #######
# parser = argparse.ArgumentParser()
# help_ = "Load h5 model trained weights"
# parser.add_argument("-w", "--weights", help=help_)
# help_ = "Use mse loss instead of binary cross entropy (default)"
# parser.add_argument("-m", "--mse", help=help_, action='store_true')
# args = parser.parse_args()
# models = (encoder, decoder)
# data = (x_test, y_test)

# VAE loss = mse_loss or xent_loss + kl_loss
# if args.mse:
#     reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
# else:
reconstruction_loss = mse(K.flatten(xin), K.flatten(outputs))

# reconstruction_loss *= nx * ny * tf.math.reduce_mean(inputs)**-1
reconstruction_loss *= nx * ny
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
K.set_value(vae.optimizer.lr, learning_rate)
K.set_value(vae.optimizer.decay, decay_rate)
vae.summary()
# plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

# Introduce Checkpoints
filepath = DataDir+'checkpoints_cond_cvae/weights.{epoch:04d}_{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True, period=10)
callback_list = [checkpoint, EarlyStopping(patience=5)]

# Resume training from previous epochs
checkpoints_path = os.listdir(DataDir+'checkpoints_cond/')
if checkpoints_path:
    vae.load_weights(DataDir+'checkpoints_cond_cvae/'+checkpoints_path[-1])
    n_epoch = int(checkpoints_path[-1][8:12])
else:
    n_epoch = 0

# train the autoencoder
vae.fit({'x_input': x_train, 'conditions': y_train, 'psf_inputs': psf_train}, batch_size=batch, epochs=epochs, initial_epoch=n_epoch, validation_data=({'x_input': x_test, 'conditions': y_test, 'psf_inputs': psf_test}, None), callbacks=callback_list)

# Save weights and models
vae.save(DataDir+'models/cond_cvae_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
vae.save_weights(DataDir+'models/cond_cvae_weights_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
encoder.save(DataDir+'models/cond_cvae_encoder_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
encoder.save_weights(DataDir+'models/cond_cvae_encoder_weights_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
decoder1.save(DataDir+'models/cond_cvae_decoder_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
decoder1.save_weights(DataDir+'models/cond_cvae_decoder_weights_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
decoder2.save(DataDir+'models/cond_cvae_psf_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')

# -------------- Training and testing sets encoding decoding -----------------

x_train_encoded = encoder.predict(x_train)
x_train_encoded = K.cast_to_floatx(x_train_encoded)
x_train_decoded = decoder1.predict(x_train_encoded[0])
x_train_decoded_conv = decoder2.predict([x_train_decoded, psf_train])

x_test_encoded = encoder.predict(x_test)
x_test_encoded = K.cast_to_floatx(x_test_encoded)
x_test_decoded = decoder1.predict(x_test_encoded[0])
x_test_decoded_conv = decoder2.predict([x_test_decoded, psf_test])

np.savetxt(DataDir+'models/cond_cvae_cosmos_encoded_xtrain_'+str(n_train)+'.txt', x_train_encoded[2])
np.savetxt(DataDir+'models/cond_cvae_cosmos_decoded_xtrain_'+str(n_train)+'.txt', np.reshape(x_train_decoded[:, :, :, 0], (x_train_decoded.shape[0], nx*ny)))
np.savetxt(DataDir+'models/cond_cvae_cosmos_decoded_psf_xtrain_'+str(n_train)+'.txt', np.reshape(x_train_decoded_conv[:, :, :, 0], (x_train_decoded_conv.shape[0], nx*ny)))

np.savetxt(DataDir+'models/cond_cvae_cosmos_encoded_xtest_'+str(n_test)+'.txt', x_test_encoded[2])
np.savetxt(DataDir+'models/cond_cvae_cosmos_decoded_xtest_'+str(n_test)+'.txt', np.reshape(x_test_decoded[:, :, :, 0], (x_test_decoded.shape[0], nx*ny)))
np.savetxt(DataDir+'models/cond_cvae_cosmos_decoded_psf_xtest_'+str(n_test)+'.txt', np.reshape(x_test_decoded_conv[:, :, :, 0], (x_test_decoded_conv.shape[0], nx*ny)))

# np.savetxt(DataDir+'cond_cvae_encoded_xtestP'+'.txt', x_test_encoded[0])
# ---------------------- GP fitting -------------------------------


def gp_fit(weights, y_train):
    """
    Learns the GP related to the weigths matrix
    Input :
    - weights : From encoder (2-D) : x_train_encoded
    - y_train : Physical parameters to interpolate

    Output :
    - model : GP model
    """
    # Set the kernel
    # kernel = GPy.kern.Matern52(input_dim=params.shape[1], variance=.1, lengthscale=.1)
    kernel = GPy.kern.Matern52(input_dim=y_train.shape[1])

    # GP Regression
    model = GPy.models.GPRegression(y_train, weights, kernel=kernel)
    model.optimize()

    # Save model
    model.save_model(DataDir+'gpmodel/gpfit_vae_cnn', compress=True, save_data=True)
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


# print('GP training ...')
# gpmodel = gp_fit(x_train_encoded[0], y_train)
# x_test_gp_encoded = gp_predict(gpmodel, y_test)
# np.savetxt(DataDir + 'models/cond_cvae_cosmos_gp_encoded_xtest_'+str(n_train)+'_'+str(n_test)+'.txt', x_test_gp_encoded)

# x_test_gp_decoded = decoder2.predict([decoder1.predict(x_test_gp_encoded), psf_test])
# np.savetxt(DataDir + 'models/cond_cvae_cosmos_gp_decoded_xtest_'+str(n_train)+'_'+str(n_test)+'.txt', np.reshape(x_test_gp_decoded, (n_test, nx*ny)))


# image_size = nx
# # -------------------- Plotting routines --------------------------

# plt.figure()
# for i in range(10):
#     plt.subplot(3, 10, i+1)
#     plt.imshow(np.reshape(x_train[i], (image_size, image_size)))
#     # plt.title('Emulated image using PCA + GP '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 10+i+1)
#     plt.imshow(np.reshape(x_train_decoded[i], (image_size, image_size)))
#     # plt.title('Simulated image using GalSim '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 20+i+1)
#     plt.imshow(np.reshape(abs(x_train_decoded[i]-x_train[i]), (image_size, image_size)))

# plt.figure()
# for i in range(10):
#     plt.subplot(3, 10, i+1)
#     plt.imshow(np.reshape(x_test[i], (image_size, image_size)))
#     # plt.title('Emulated image using PCA + GP '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 10+i+1)
#     plt.imshow(np.reshape(x_test_decoded[i], (image_size, image_size)))
#     # plt.title('Simulated image using GalSim '+str(i))
#     # plt.colorbar()
#     plt.subplot(3, 10, 20+i+1)
#     plt.imshow(np.reshape(abs(x_test_decoded[i]-x_test[i]), (image_size, image_size)))

# plt.show()

# PlotScatter = False
# if PlotScatter:

#     w1 = 1
#     w2 = 2
#     # display a 2D plot of latent space (just 2 dimensions)
#     plt.figure(figsize=(6, 6))

#     x_train_encoded = encoder.predict(x_train)
#     plt.scatter(x_train_encoded[0][:, w1], x_train_encoded[0][:, w2], c=y_train[:, 0], cmap='spring')
#     plt.colorbar()

#     x_test_encoded = encoder.predict(x_test)
#     plt.scatter(x_test_encoded[0][:, w1], x_test_encoded[0][:, w2], c=y_test[:, 0], cmap='copper')
#     plt.colorbar()
#     # plt.title(fileOut)
#     plt.savefig('cond_cvae_Scatter_z'+'.png')

#     # Plot losses
#     n_epochs = np.arange(1, epochs+1)
#     train_loss = vae.history.history['loss']
#     val_loss = np.ones_like(train_loss)
#     fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
#     ax.plot(n_epochs, train_loss, '-', lw=1.5)
#     ax.plot(n_epochs, val_loss, '-', lw=1.5)
#     ax.set_ylabel('loss')
#     ax.set_xlabel('epochs')
#     ax.legend(['train loss', 'val loss'])
#     plt.tight_layout()

# plt.show()
