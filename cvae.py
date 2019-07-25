#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 2019

@author: cguilloteau
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
# from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
# import argparse
import os
import h5py
import network_params as netparam

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def rescale(params):
    """
    Rescales parameters between -1 and 1.
    Input :
    - params : physical parameters
    Outputs :
    - params_new : rescaled parameters
    - theta_mean, theta_mult : rescaling factors
    """
    theta_mean = np.mean(params, axis=0)
    theta_mult = np.max(params - theta_mean, axis=0)
    return (params - theta_mean) * theta_mult**-1, theta_mean, theta_mult


def sampling(args):
    """
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    Input :
     - args (tensor): mean and log of variance of Q(z|X)
    Output :
     - z (tensor): sampled latent vector
    """
    epsilon_mean = netparam.epsilon_mean
    epsilon_std = netparam.epsilon_std

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim), mean=epsilon_mean, stddev=epsilon_std)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def train_cvae(latent_dim, filename_train_gal, filename_test_gal):
    # ------------------------ Parameters ---------------------------------------
    DataDir = netparam.DataDir

    batch = netparam.batch
    kernel_size = netparam.kernel_size
    n_conv = netparam.n_conv
    filters = netparam.filters
    interm_dim = netparam.interm_dim
    # latent_dim = netparam.latent_dim
    epochs = netparam.epochs

    # epsilon_mean = netparam.epsilon_mean
    # epsilon_std = netparam.epsilon_std

    learning_rate = netparam.learning_rate
    decay_rate = netparam.decay_rate

    # ------------------------ Loading and rescaling ----------------------------

    # Load training/testing set
    x_train = np.array(h5py.File(DataDir + 'output_tests/' + filename_train_gal + '.hdf5', 'r')['galaxies'])[:, :32, :32]
    ntrain = x_train.shape[0]
    x_test = np.array(h5py.File(DataDir + 'output_tests/' + filename_test_gal + '.hdf5', 'r')['galaxies'])[:, :32, :32]

    # Rescaling
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_test = (x_test - xmin) / xmax

    x_train = K.cast_to_floatx(x_train)
    x_test = K.cast_to_floatx(x_test)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

    x_train = x_train.astype('float32')  # / 255
    x_test = x_test.astype('float32')  # / 255

    # ------------------------ Build models -----------------------------------

    input_shape = (image_size, image_size, 1)

    # VAE model = encoder + decoder
    # Build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Convolutional layers
    for i in range(n_conv):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # Shape info needed to build decoder model
    shape = K.int_shape(x)

    # Dense layers and generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(interm_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # Print encoder model and save png
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # Build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    # Dense layers
    x = Dense(interm_dim, activation='relu')(latent_inputs)
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # Convolutional layers
    for i in range(n_conv):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    # Output
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # Instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    # Print encoder model and save png
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # Instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    models = (encoder, decoder)

    # -------------------- Loss function and optimizer -----------------------

    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size

    # KL divergence
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile and set learning rate and decay rate
    vae.compile(optimizer='rmsprop')
    K.set_value(vae.optimizer.lr, learning_rate)
    K.set_value(vae.optimizer.decay, decay_rate)

    # Print vae model and save png
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    # ------------------------- Training -----------------------------------------
    vae.fit(x_train, batch_size=batch, epochs=epochs, validation_data=(x_test, None))

    # Save weights and models
    vae.save(DataDir+'cvae_model_galsim'+str(ntrain)+'.h5')
    vae.save_weights(DataDir+'cvae_weights_galsim'+str(ntrain)+'.h5')
    encoder.save(DataDir+'cvae_encoder_model_galsim'+str(ntrain)+'.h5')
    encoder.save_weights(DataDir+'cvae_encoder_weights_galsim'+str(ntrain)+'.h5')

    filename_decoder = 'cvae_decoder_model_galsim'+str(ntrain)+'.h5'
    decoder.save(DataDir+filename_decoder)
    decoder.save_weights(DataDir+'cvae_decoder_weights_galsim'+str(ntrain)+'.h5')

    # -------------- Training and testing sets encoding decoding -----------------

    x_train_encoded = encoder.predict(x_train)
    x_train_encoded = K.cast_to_floatx(x_train_encoded)

    filename_train_encoded = 'cvae_encoded_xtrain_512_5'+str(ntrain)+'.txt'
    np.savetxt(DataDir+filename_train_encoded, x_train_encoded[2])

    return filename_decoder, filename_train_encoded


def main():
    # ------------------------ Parameters ---------------------------------------
    DataDir = netparam.DataDir

    batch = netparam.batch
    kernel_size = netparam.kernel_size
    n_conv = netparam.n_conv
    filters = netparam.filters
    interm_dim = netparam.interm_dim
    latent_dim = netparam.latent_dim
    epochs = netparam.epochs

    # epsilon_mean = netparam.epsilon_mean
    # epsilon_std = netparam.epsilon_std

    learning_rate = netparam.learning_rate
    decay_rate = netparam.decay_rate

    # ------------------------ Loading and rescaling ----------------------------

    # Load training/testing set
    x_train = np.array(h5py.File(DataDir + 'output_tests/training_512_5.hdf5', 'r')['galaxies'])[:, :32, :32]
    x_test = np.array(h5py.File(DataDir + 'output_tests/test_64_5_testing.hdf5', 'r')['galaxies'])[:, :32, :32]

    # print(x_train.shape, 'train sequences')
    # print(x_test.shape, 'test sequences')

    # Rescaling
    xmin = np.min(x_train)
    xmax = np.max(x_train) - xmin
    x_train = (x_train - xmin) / xmax
    x_test = (x_test - xmin) / xmax

    x_train = K.cast_to_floatx(x_train)
    x_test = K.cast_to_floatx(x_test)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

    x_train = x_train.astype('float32')  # / 255
    x_test = x_test.astype('float32')  # / 255

    # ------------------------ Build models -----------------------------------

    input_shape = (image_size, image_size, 1)

    # VAE model = encoder + decoder
    # Build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # Convolutional layers
    for i in range(n_conv):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # Shape info needed to build decoder model
    shape = K.int_shape(x)

    # Dense layers and generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(interm_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # Use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # Print encoder model and save png
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # Build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    # Dense layers
    x = Dense(interm_dim, activation='relu')(latent_inputs)
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # Convolutional layers
    for i in range(n_conv):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    # Output
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # Instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    # Print encoder model and save png
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # Instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    models = (encoder, decoder)

    # -------------------- Loss function and optimizer -----------------------

    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size

    # KL divergence
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile and set learning rate and decay rate
    vae.compile(optimizer='rmsprop')
    K.set_value(vae.optimizer.lr, learning_rate)
    K.set_value(vae.optimizer.decay, decay_rate)

    # Print vae model and save png
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    # ------------------------- Training -----------------------------------------
    vae.fit(x_train, batch_size=batch, epochs=epochs, validation_data=(x_test, None))

    # Save weights and models
    vae.save(DataDir+'cvae_model_galsim.h5')
    vae.save_weights(DataDir+'cvae_weights_galsim.h5')
    encoder.save(DataDir+'cvae_encoder_model_galsim.h5')
    encoder.save_weights(DataDir+'cvae_encoder_weights_galsim.h5')
    decoder.save(DataDir+'cvae_decoder_model_galsim.h5')
    decoder.save_weights(DataDir+'cvae_decoder_weights_galsim.h5')

    # -------------- Training and testing sets encoding decoding -----------------

    x_train_encoded = encoder.predict(x_train)
    x_train_encoded = K.cast_to_floatx(x_train_encoded)
    x_train_decoded = decoder.predict(x_train_encoded[0])

    x_test_encoded = encoder.predict(x_test)
    x_test_encoded = K.cast_to_floatx(x_test_encoded[0])
    x_test_decoded = decoder.predict(x_test_encoded)

    np.savetxt(DataDir+'cvae_encoded_xtrain_512_5.txt', x_train_encoded[2])
    np.savetxt(DataDir+'cvae_decoded_xtrain_512_5.txt', np.reshape(x_train_decoded[:, :, :, 0], (x_train_decoded.shape[0], image_size**2)))
    # np.savetxt(DataDir+'cvae_encoded_xtestP'+'.txt', x_test_encoded[0])

    # ---------------------------- Plotting routines -----------------------------

    plt.figure()
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(np.reshape(x_train[i], (image_size, image_size)))
        # plt.title('Emulated image using PCA + GP '+str(i))
        # plt.colorbar()
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(np.reshape(x_train_decoded[i], (image_size, image_size)))
        # plt.title('Simulated image using GalSim '+str(i))
        # plt.colorbar()

    plt.figure()
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(np.reshape(x_test[i], (image_size, image_size)))
        # plt.title('Emulated image using PCA + GP '+str(i))
        # plt.colorbar()
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(np.reshape(x_test_decoded[i], (image_size, image_size)))
        # plt.title('Simulated image using GalSim '+str(i))
        # plt.colorbar()
    plt.show()

    PlotScatter = True
    if PlotScatter:

        # Load ans rescale physical parameters
        y_train = np.loadtxt(DataDir + 'lhc_512_5.txt')
        y_test = np.loadtxt(DataDir + 'lhc_64_5_testing.txt')
        y_train, ymean, ymult = rescale(y_train)
        y_test = (y_test - ymean) * ymult**-1

        # Choose weights to plor
        w1 = 1
        w2 = 2
        # display a 2D plot of latent space (just 2 dimensions)
        plt.figure(figsize=(6, 6))

        x_train_encoded = encoder.predict(x_train)
        plt.scatter(x_train_encoded[0][:, w1], x_train_encoded[0][:, w2], c=y_train[:, 0], cmap='summer')
        plt.colorbar()

        x_test_encoded = encoder.predict(x_test)
        plt.scatter(x_test_encoded[0][:, w1], x_test_encoded[0][:, w2], c=y_test[:, 0], cmap='cool')
        plt.colorbar()
        # plt.title(fileOut)
        plt.savefig('cvae_Scatter_z'+'.png')

        # Plot losses
        n_epochs = np.arange(1, epochs+1)
        train_loss = vae.history.history['loss']
        val_loss = np.ones_like(train_loss)
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
        ax.plot(n_epochs, train_loss, '-', lw=1.5)
        ax.plot(n_epochs, val_loss, '-', lw=1.5)
        ax.set_ylabel('loss')
        ax.set_xlabel('epochs')
        ax.legend(['train loss', 'val loss'])
        plt.tight_layout()
    plt.show()
