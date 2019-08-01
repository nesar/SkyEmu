from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda, GaussianNoise
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
# from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import GPy

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


# Convolution with PSF function
def psf_convolve(img, psf):
    output = np.fft.ifft2(np.fft.fft2(img[0], norm='ortho')*psf, norm='ortho')
    return np.expand_dims(output)


# Add some poisson noise function
def add_poisson_noise(img):
    batch = K.shape(img)[0]
    dim = K.shape(img)[1]
    noise = K.random_normal(shape=(batch, dim))
    return img + noise * np.sqrt(img)


# Plotting routine
def plot_results(x_train, x_test):
    plt.figure()
    for i in range(5):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_train[0, i])
        plt.subplot(3, 5, 5+i+1)
        plt.imshow(x_train[2, i])
        plt.subplot(3, 5, 10+i+1)
        plt.imshow(x_train[3, i])
    plt.figure()
    for i in range(10):
        plt.subplot(3, 5, i+1)
        plt.imshow(x_test[0, i])
        plt.subplot(3, 5, 5+i+1)
        plt.imshow(x_test[2, i])
        plt.subplot(3, 5, 10+i+1)
        plt.imshow(x_test[3, i])
    plt.show()


# ----------------------------- LOAD PARAMETERS -----------------------------

# network parameters
DataDir = '../Data/Cosmos/'

n_train = 2048
n_test = 128
nx = 64
ny = 64

epsilon_mean = 0.
epsilon_std = 1.e-4

input_shape = (nx, ny, 1)
batch = 32
kernel_size = 4
n_conv = 0
filters = 16
interm_dim = 128
latent_dim = 20
epochs = 1000

learning_rate = 1e-4
decay_rate = 1e-1

stddev = 1

# ------------------------------ LOAD DATA ----------------------------------

# Load training set images and rescale fluxes
path = DataDir + 'data/cosmos_real_train_'+str(n_train)+'.hdf5'
f = h5py.File(path, 'r')
x_train = np.array(f['real galaxies'])
xmin = np.min(x_train)
xmax = np.max(x_train) - xmin
x_train = (x_train - xmin) / xmax
x_train = np.reshape(x_train, (n_train, nx*ny))

# Load testing set parametric images
x_train_parametric = (np.array(f['parametric galaxies']) - xmin) / xmax
x_train_parametric = np.reshape(x_train_parametric, (n_train, nx*ny))

# Load training set parameters and rescale
y_train = np.array(f['parameters'])
ymean = np.mean(y_train, axis=0)
ymax = np.mean(y_train - ymean, axis=0)
y_train = (y_train - ymean) * ymax**-1

# Load training set psfs
psf_train = np.array(f['psf'], dtype=np.complex)
f.close()

# Load testing set and rescale fluxes
path = DataDir + 'data/cosmos_real_test_'+str(n_test)+'.hdf5'
f = h5py.File(path, 'r')
x_test = np.array(f['real galaxies'])
x_test = (x_test - xmin) / xmax
x_test = np.reshape(x_test, (n_test, nx*ny))

# Load testing set parameters and rescale
y_test = np.array(f['parameters'])
y_test = (y_test - ymean) * ymax**-1

# Load testing set parametric images
x_test_parametric = (np.array(f['parametric galaxies']) - xmin) / xmax
x_test_parametric = np.reshape(x_test_parametric, (n_test, nx*ny))

# Load training set psfs
psf_test = np.array(f['psf'], dtype=np.complex)
f.close()

# Cast to float, reshaping, ...
x_train = K.cast_to_floatx(x_train)
x_test = K.cast_to_floatx(x_test)
x_train = np.reshape(x_train, [-1, nx, ny, 1])
x_test = np.reshape(x_test, [-1, nx, ny, 1])
x_train = x_train.astype('float32')  # / 255
x_test = x_test.astype('float32')  # / 255

# ------------------- VAE model = encoder + decoder -------------------------

# ******** ENCODER ********
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(n_conv):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(interm_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# ******** DECODER ********
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(interm_dim, activation='relu')(latent_inputs)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(n_conv):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

x_dec = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)

x = Lambda(psf_convolve, output_shape=(1, nx, ny))(x_dec, psf)

outputs = GaussianNoise(stddev)(x)

# instantiate decoder model
decoder_deconv = Model(latent_inputs, x_dec, name='decoder_deconvolved')
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

models = (encoder, decoder)

# ******** LOSS AND OPTIMIZER ********

# VAE loss = mse_loss or xent_loss + kl_loss
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

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

# ******** TRAINING ********
# train the autoencoder
vae.fit(x_train, batch_size=batch, epochs=epochs, validation_data=(x_test, None))

# Save weights and models
vae.save(DataDir+'models/cnn_vae_model_cosmos.h5')
vae.save_weights(DataDir+'cnn_vae_weights_cosmos.h5')

encoder.save(DataDir+'models/cnn_vae_encoder_model_galsim.h5')
encoder.save_weights(DataDir+'models/cnn_vae_encoder_weights_cosmos.h5')

decoder_deconv.save(DataDir+'models/cnn_vae_decoder_deconv_model_cosmos.h5')
decoder_deconv.save_weights(DataDir+'models/cnn_vae_decoder_deconv_weights_cosmos.h5')

decoder.save(DataDir+'models/cnn_vae_decoder_model_cosmos.h5')
decoder.save_weights(DataDir+'models/cnn_vae_decoder_weights_cosmos.h5')

# --------------------------------- Saving ---------------------------------

x_train_encoded = encoder.predict(x_train)
x_train_encoded = K.cast_to_floatx(x_train_encoded)
x_train_decoded_deconv = decoder_deconv(x_train_encoded[0])
x_train_decoded = decoder.predict(x_train_encoded[0])

x_test_encoded = encoder.predict(x_test)
x_test_encoded = K.cast_to_floatx(x_test_encoded[0])
x_test_decoded_deconv = decoder_deconv(x_train_encoded[0])
x_test_decoded = decoder.predict(x_test_encoded)

f = h5py.File(DataDir + 'results/x_train.hdf5', 'w')
f.create_dataset('encoded', data=x_train_encoded)
f.create_dataset('decoded_deconv', data=x_train_decoded_deconv)
f.create_dataset('decoded', data=x_train_decoded)
f.close()

f = h5py.File(DataDir + 'results/x_test.hdf5', 'w')
f.create_dataset('encoded', data=x_test_encoded)
f.create_dataset('decoded_deconv', data=x_test_decoded_deconv)
f.create_dataset('decoded', data=x_test_decoded)
f.close()

# --------------------------------- Plot ---------------------------------

plot = True
if plot:
    plot_results([x_train, x_train_encoded, x_train_decoded_deconv, x_train_decoded], [x_test, x_test_encoded, x_test_decoded_deconv, x_test_decoded])
