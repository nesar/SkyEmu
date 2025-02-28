from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
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
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
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

    filename = os.path.join(model_name, "digits_over_latent.png")
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
    plt.show()


################ edits ########################
# MNIST dataset
# (x_train1, y_train1), (x_test1, y_test1) = mnist.load_data()
# image_size1 = x_train1.shape[1]
# x_train1 = np.reshape(x_train1, [-1, image_size1, image_size1, 1])
# x_test1 = np.reshape(x_test1, [-1, image_size1, image_size1, 1])

# Load training/testing set
DataDir = '../Data/'
x_train = np.array(h5py.File(DataDir + 'Cosmos/data/cosmos_train_512.hdf5', 'r')['galaxies'])
x_test = np.array(h5py.File(DataDir + 'Cosmos/data/cosmos_test_64.hdf5', 'r')['galaxies'])

# y_train = np.loadtxt(DataDir + 'lhc_512_5.txt')
# y_test = np.loadtxt(DataDir + 'lhc_64_5_testing.txt')

# x_train = Trainfiles[:, num_para+2:]
# x_test = Testfiles[:, num_para+2:]
# y_train = Trainfiles[:, 0: num_para]
# y_test =  Testfiles[:, 0: num_para]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
# print(y_train.shape, 'train sequences')
# print(y_test.shape, 'test sequences')


# Rescaling
xmin = np.min(x_train)
xmax = np.max(x_train) - xmin
x_train = (x_train - xmin) / xmax
x_test = (x_test - xmin) / xmax

# y_train, ymean, ymult = rescale(y_train)
# y_test = (y_test - ymean) * ymult**-1

# print(y_train)
# print('----')
# print(y_test)

# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*, x_train.shape[2]))
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

print(x_train.shape)
print(x_test.shape)

# print(x_train1.shape)
# print(x_test1.shape)

x_train = K.cast_to_floatx(x_train)
x_test = K.cast_to_floatx(x_test)

################################################
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])

################################################


x_train = x_train.astype('float32')  # / 255
x_test = x_test.astype('float32')  # / 255

# network parameters
input_shape = (image_size, image_size, 1)
batch = 32
kernel_size = 4
n_conv = 2
filters = 16
interm_dim = 128
latent_dim = 20
epochs = 2

epsilon_mean = 0.
epsilon_std = 1e-4

learning_rate = 1e-4
decay_rate = 1e-1

# VAE model = encoder + decoder
# build encoder model
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
plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
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

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)



# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

####### main function #######
# parser = argparse.ArgumentParser()
# help_ = "Load h5 model trained weights"
# parser.add_argument("-w", "--weights", help=help_)
# help_ = "Use mse loss instead of binary cross entropy (default)"
# parser.add_argument("-m", "--mse", help=help_, action='store_true')
# args = parser.parse_args()
models = (encoder, decoder)
# data = (x_test, y_test)

# VAE loss = mse_loss or xent_loss + kl_loss
# if args.mse:
#     reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
# else:
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
K.set_value(vae.optimizer.lr, learning_rate)
K.set_value(vae.optimizer.decay, decay_rate)
vae.summary()
plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

# if args.weights:
#     vae.load_weights(args.weights)
# else:
# train the autoencoder
vae.fit(x_train, batch_size=batch, epochs=epochs, validation_data=(x_test, None))
# vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

vae.save_weights('vae_cnn_galsim_cosmos.h5')

# plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")
# Saving
# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)

x_train_encoded = K.cast_to_floatx(x_train_encoded)

x_train_decoded = decoder.predict(x_train_encoded[0])

x_test_encoded = encoder.predict(x_test)

x_test_encoded = K.cast_to_floatx(x_test_encoded[0])

x_test_decoded = decoder.predict(x_test_encoded)

np.savetxt(DataDir+'cvae_encoded_xtrain_cosmos.txt', x_train_encoded[0])
np.savetxt(DataDir+'cvae_encoded_xtest_cosmos.txt', x_test_encoded[0])

np.savetxt(DataDir+'cvae_decoded_xtrain_cosmos.txt', np.reshape(x_train_decoded, (x_train_decoded.shape[0], x_train_decoded.shape[1]*x_train_decoded.shape[2])))
np.savetxt(DataDir+'cvae_decoded_xtest_cosmos.txt', np.reshape(x_test_decoded, (x_test_decoded.shape[0], x_test_decoded.shape[1]*x_test_decoded.shape[2])))

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
    model.save_model('../Data/GPmodel/gpfit_cvae', compress=True, save_data=True)
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


# gpmodel = gp_fit(x_train_encoded[0], y_train)

# x_test_encoded = gp_predict(gpmodel, y_test)

# np.savetxt(DataDir + 'x_test_encoded_64_5.txt', x_test_encoded)

# x_test_decoded = decoder.predict(x_test_encoded)


# -------------------- Plotting routines --------------------------

plt.figure()
for i in range(10):
    plt.subplot(3, 10, i+1)
    plt.imshow(np.reshape(x_train[i], (image_size, image_size)))
    # plt.title('Emulated image using PCA + GP '+str(i))
    # plt.colorbar()
    plt.subplot(3, 10, 10+i+1)
    plt.imshow(np.reshape(x_train_decoded[i], (image_size, image_size)))
    # plt.title('Simulated image using GalSim '+str(i))
    # plt.colorbar()
    plt.subplot(3, 10, 20+i+1)
    plt.imshow(np.reshape(abs(x_train_decoded[i]-x_train[i]), (image_size, image_size)))

plt.figure()
for i in range(10):
    plt.subplot(3, 10, i+1)
    plt.imshow(np.reshape(x_test[i], (image_size, image_size)))
    # plt.title('Emulated image using PCA + GP '+str(i))
    # plt.colorbar()
    plt.subplot(3, 10, 10+i+1)
    plt.imshow(np.reshape(x_test_decoded[i], (image_size, image_size)))
    # plt.title('Simulated image using GalSim '+str(i))
    # plt.colorbar()
    plt.subplot(3, 10, 20+i+1)
    plt.imshow(np.reshape(abs(x_test_decoded[i]-x_test[i]), (image_size, image_size)))

plt.show()

PlotScatter = False
if PlotScatter:

    w1 = 1
    w2 = 2
    # display a 2D plot of latent space (just 2 dimensions)
    plt.figure(figsize=(6, 6))

    x_train_encoded = encoder.predict(x_train)
    plt.scatter(x_train_encoded[0][:, w1], x_train_encoded[0][:, w2], c=y_train[:, 0], cmap='spring')
    plt.colorbar()

    x_test_encoded = encoder.predict(x_test)
    plt.scatter(x_test_encoded[0][:, w1], x_test_encoded[0][:, w2], c=y_test[:, 0], cmap='copper')
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
