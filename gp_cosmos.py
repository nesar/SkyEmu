import numpy as np
import GPy
import h5py
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import load_model, Model
import tensorflow as tf


# Convolution with PSF function
def psf_convolve(args):
    img, psf = args
    imgfft = tf.spectral.rfft2d(img[:, :, :, 0])
    psffft = tf.spectral.rfft2d(psf[:, :, :, 0])
    convfft = tf.spectral.irfft2d(imgfft * psffft)
    h = tf.expand_dims(convfft, axis=-1)
    return h


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
    kernel = GPy.kern.Matern52(input_dim=y_train.shape[1], variance=.001, lengthscale=.1)
    # kernel = GPy.kern.Matern52(input_dim=y_train.shape[1])

    # GP Regression
    model = GPy.models.GPRegression(y_train, weights, kernel=kernel)
    model.optimize()

    # Save model
    model.save_model('../Data/Cosmos/gpmodel/gpfit_cvae_cosmos_train_'+str(n_train), compress=True, save_data=True)
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

# network parameters
DataDir = '../Data/Cosmos/'

n_train = 4096
n_test = 256
nx = 64
ny = 64
input_shape = (nx, ny, 1)

# Load training set images and rescale fluxes
path = DataDir + 'data/cosmos_real_trainingset_train_'+str(n_train)+'_test_'+str(n_test)+'.hdf5'
f = h5py.File(path, 'r')
x_train = np.array(f['real galaxies'])
xmin = np.min(x_train)
xmax = np.max(x_train) - xmin
x_train = (x_train - xmin) / xmax
x_train = np.reshape(x_train, (n_train, nx*ny))

# Load testing set parametric images
# x_train_parametric = (np.array(f['parametric galaxies']) - xmin) / xmax
# x_train_parametric = np.reshape(x_train_parametric, (n_train, nx*ny))

# Load training set parameters and rescale
y_train = np.array(f['parameters'])
ymean = np.mean(y_train, axis=0)
ymax = np.max(y_train - ymean, axis=0)
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

# Load testing set parametric images
# x_test_parametric = (np.array(f['parametric galaxies']) - xmin) / xmax
# x_test_parametric = np.reshape(x_test_parametric, (n_test, nx*ny))

# Load training set psfs
psf_test = np.fft.fftshift(np.array(f['psf']))
f.close()

# Cast to float, reshaping, ...
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

x_train_encoded = np.loadtxt(DataDir+'models/cvae_cosmos_encoded_xtrain_'+str(n_train)+'.txt')
decoder1 = load_model(DataDir+'models/cvae_decoder_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')
# decoder2 = load_model(DataDir+'models/cvae_psf_model_cosmos_'+str(n_train)+'_train_'+str(n_test)+'_test.h5')

dec_inputs2 = Input(shape=input_shape, name='dec_inputs2')
psf_inputs = Input(shape=input_shape, name='psf_inputs')
outputs2 = Lambda(psf_convolve, output_shape=input_shape)([dec_inputs2, psf_inputs])
decoder2 = Model([dec_inputs2, psf_inputs], outputs2)

print('GP training ...')
gpmodel = gp_fit(x_train_encoded, y_train)
print('GP pediction ...')
x_test_gp_encoded = gp_predict(gpmodel, y_test)
np.savetxt(DataDir + 'models/cvae_cosmos_gp_encoded_xtest_'+str(n_train)+'_'+str(n_test)+'.txt', x_test_gp_encoded)

print('Decoding ...')
x_test_gp_decoded = decoder2.predict([decoder1.predict(x_test_gp_encoded), psf_test])
np.savetxt(DataDir + 'models/cvae_cosmos_gp_decoded_xtest_'+str(n_train)+'_'+str(n_test)+'.txt', np.reshape(x_test_gp_decoded, (n_test, nx*ny)))
