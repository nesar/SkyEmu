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

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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


# network parameters
DataDir = '../Data/Cosmos/'

n_train = 20000
n_test = 2000
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
f.close()

# Cast to float, reshaping, ...
x_train = K.cast_to_floatx(x_train)
x_test = K.cast_to_floatx(x_test)
y_train = K.cast_to_floatx(y_train)
y_test = K.cast_to_floatx(y_test)

x_train = np.reshape(x_train, [-1, nx*ny, ])
x_test = np.reshape(x_test, [-1, nx*ny, ])
y_train = np.reshape(y_train, [-1, nparams, ])
y_test = np.reshape(y_test, [-1, nparams, ])

x_train = x_train.astype('float32')  # / 255
x_test = x_test.astype('float32')  # / 255
y_train = y_train.astype('float32')  # / 255
y_test = y_test.astype('float32')  # / 255

# network parameters
input_shape = (nx, ny, 1)
batch = 32
# kernel_size = 4
# n_conv = 2
# filters = 8
# shape = (60, 60, 1)
interm_dim1 = 2048
interm_dim2 = 1024
interm_dim3 = 512
interm_dim4 = 256
interm_dim5 = 128
interm_dim6 = 64
latent_dim = 32
epochs = 10
drop = 0.1
l1_ = 0.01
l2_ = 0.01

epsilon_mean = 0.
epsilon_std = 1e-5

learning_rate = 1e-4
decay_rate = 1e-1

# VAE model = encoder + decoder
# ENCODER

# Inputs
xin = Input(shape=(nx*ny, ), name='x_input')
cond = Input(shape=(nparams, ), name='conditions')
inputs = concatenate([xin, cond])

# Dense layers
x = Dense(interm_dim1, activation='relu')(inputs)
x = Dense(interm_dim2, activation='relu')(x)
x = Dropout(drop)(x)
x = Dense(interm_dim3, activation='relu')(x)
x = Dense(interm_dim4, activation='relu')(x)
x = Dropout(drop)(x)
x = Dense(interm_dim5, activation='relu')(x)
x = Dense(interm_dim6, activation='relu')(x)
# Mean and std
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Sampling
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# DECODER
latent_inputs1 = concatenate([z, cond])

# Dense layers
decoder_dense1 = Dense(interm_dim6, activation='relu')
decoder_drop = Dropout(drop)
decoder_dense2 = Dense(interm_dim5, activation='relu')
decoder_dense3 = Dense(interm_dim4, activation='relu')
decoder_dense4 = Dense(interm_dim3, activation='relu')
decoder_dense5 = Dense(interm_dim2, activation='relu')
decoder_dense6 = Dense(interm_dim1, activation='relu')
decoder_final = Dense(nx*ny, activation='relu')

x1 = decoder_dense1(latent_inputs1)
x1 = decoder_dense2(x1)
x1 = decoder_drop(x1)
x1 = decoder_dense3(x1)
x1 = decoder_dense4(x1)
x1 = decoder_drop(x1)
x1 = decoder_dense5(x1)
x1 = decoder_dense6(x1)
outputs_d = decoder_final(x1)

# Instantiate encoder model
encoder = Model([xin, cond], z, name='encoder')
encoder.summary()

# Instantiate decoder model
zin = Input(shape=(latent_dim+nparams, ), name='z_sampling')
x1 = decoder_dense1(zin)
x1 = decoder_dense2(x1)
x1 = decoder_drop(x1)
x1 = decoder_dense3(x1)
x1 = decoder_dense4(x1)
x1 = decoder_drop(x1)
x1 = decoder_dense5(x1)
x1 = decoder_dense6(x1)
outputs_dec = decoder_final(x1)
decoder = Model(zin, outputs_dec, name='decoder')
decoder.summary()

# instantiate VAE model
vae = Model([xin, cond], outputs_d, name='vae')


# --------------------------------------------------------------------------------------------------------
# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = nx*ny*mse(y_pred, y_true)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)
    return recon + kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1))

def recon_loss(y_true, y_pred):
	return nx*ny*mse(y_pred, y_true)

vae.compile(optimizer='rmsprop', loss=vae_loss, metrics = [KL_loss, recon_loss])
K.set_value(vae.optimizer.lr, learning_rate)
K.set_value(vae.optimizer.decay, decay_rate)
# vae.summary()

# Introduce Checkpoints
filepath = DataDir+'checkpoints_cond/weights.{epoch:04d}_{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, save_weights_only=True, period=10)
callback_list = [checkpoint, EarlyStopping(patience=5)]

# Resume training from previous epochs
checkpoints_path = os.listdir(DataDir+'checkpoints_cond/')
if checkpoints_path:
    vae.load_weights(DataDir+'checkpoints_cond/'+checkpoints_path[-1])
    n_epoch = int(checkpoints_path[-1][8:12])
else:
    n_epoch = 0

# train the autoencoder
vae.fit({'x_input': x_train, 'conditions': y_train}, x_train, batch_size=batch, epochs=epochs, initial_epoch=n_epoch, validation_data=({'x_input': x_test, 'conditions': y_test}, x_test), callbacks=callback_list)

# Save weights and models
vae.save(DataDir+'models/cond_vae_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
vae.save_weights(DataDir+'models/cond_vae_weights_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
encoder.save(DataDir+'models/cond_vae_encoder_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
encoder.save_weights(DataDir+'models/cond_vae_encoder_weights_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
decoder.save(DataDir+'models/cond_vae_decoder_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
decoder.save_weights(DataDir+'models/cond_vae_decoder_weights_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')

# -------------- Training and testing sets encoding decoding -----------------

x_train_encoded = encoder.predict(x_train)
x_train_encoded = K.cast_to_floatx(x_train_encoded)
x_train_decoded = decoder1.predict(x_train_encoded[0])
x_train_decoded_conv = decoder2.predict([x_train_decoded, psf_train])

x_test_encoded = encoder.predict(x_test)
x_test_encoded = K.cast_to_floatx(x_test_encoded)
x_test_decoded = decoder1.predict(x_test_encoded[0])
x_test_decoded_conv = decoder2.predict([x_test_decoded, psf_test])

np.savetxt(DataDir+'models/cond_vae_cosmos_encoded_xtrain_'+str(n_train)+'.txt', x_train_encoded[2])
np.savetxt(DataDir+'models/cond_vae_cosmos_decoded_xtrain_'+str(n_train)+'.txt', np.reshape(x_train_decoded[:, :, :, 0], (x_train_decoded.shape[0], nx*ny)))
np.savetxt(DataDir+'models/cond_vae_cosmos_decoded_psf_xtrain_'+str(n_train)+'.txt', np.reshape(x_train_decoded_conv[:, :, :, 0], (x_train_decoded_conv.shape[0], nx*ny)))

np.savetxt(DataDir+'models/cond_vae_cosmos_encoded_xtest_'+str(n_test)+'.txt', x_test_encoded[2])
np.savetxt(DataDir+'models/cond_vae_cosmos_decoded_xtest_'+str(n_test)+'.txt', np.reshape(x_test_decoded[:, :, :, 0], (x_test_decoded.shape[0], nx*ny)))
np.savetxt(DataDir+'models/cond_vae_cosmos_decoded_psf_xtest_'+str(n_test)+'.txt', np.reshape(x_test_decoded_conv[:, :, :, 0], (x_test_decoded_conv.shape[0], nx*ny)))























