import numpy as np
import h5py
import os
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras import losses

import matplotlib.pyplot as plt
import keras.backend as K

import params_debug as params

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

############### Setting same float, random seeds ##############

np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
K.set_floatx('float32')

###################### PARAMETERS ##############################

original_dim = params.original_dim  # 2549
intermediate_dim3 = params.intermediate_dim3  # 1600
intermediate_dim2 = params.intermediate_dim2  # 1024
intermediate_dim1 = params.intermediate_dim1  # 512
intermediate_dim0 = params.intermediate_dim0  # 256
intermediate_dim = params.intermediate_dim  # 256
latent_dim = params.latent_dim  # 10

# ClID = params.ClID
num_train = params.num_train  # 512
num_test = params.num_test  # 32
num_para = params.num_para  # 5


batch_size = params.batch_size  # 8
num_epochs = params.num_epochs  # 100
epsilon_mean = params.epsilon_mean  # 1.0
epsilon_std = params.epsilon_std  # 1.0
learning_rate = params.learning_rate  # 1e-3
decay_rate = params.decay_rate  # 0.0

noise_factor = params.noise_factor  # 0.00

######################## I/O ##################################

DataDir = params.DataDir
PlotsDir = params.PlotsDir
ModelDir = params.ModelDir

fileOut = params.fileOut

# Load training/testing set

x_train = np.array(h5py.File(DataDir + '/output_tests/training_512_5.hdf5', 'r')['galaxies'])
x_test = np.array(h5py.File(DataDir + '/output_tests/test_64_5.hdf5', 'r')['galaxies'])

y_train = np.loadtxt(DataDir + 'lhc_512_5.txt')
y_test = np.loadtxt(DataDir + 'lhc_64_5_test.txt')

# x_train = Trainfiles[:, num_para+2:]
# x_test = Testfiles[:, num_para+2:]
# y_train = Trainfiles[:, 0: num_para]
# y_test =  Testfiles[:, 0: num_para]

# print(x_train.shape, 'train sequences')
# print(x_test.shape, 'test sequences')
# print(y_train.shape, 'train sequences')
# print(y_test.shape, 'test sequences')


# Rescaling
xmax = np.max(x_train)
x_train /= xmax
x_test /= xmax

y_train, ymean, ymult = rescale(y_train)
y_test = (y_test - ymean) * ymult**-1

# print(y_train)
# print('----')
# print(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

print(x_train.shape)
print(x_test.shape)

x_train = K.cast_to_floatx(x_train)
x_test = K.cast_to_floatx(x_test)

# Architecture

# inputs = Input(shape=(original_dim,))


input_shape = (x_train.shape[1], x_train.shape[1])

inputs = Input(shape= input_shape, name='encoder_input')
x = inputs
for i in range(2):
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




h_q3 = Dense(intermediate_dim3, activation='linear')(inputs)  # ADDED intermediate layer
h_q2 = Dense(intermediate_dim2, activation='linear')(h_q3)  # ADDED intermediate layer
h_q1 = Dense(intermediate_dim1, activation='linear')(h_q2)  # ADDED intermediate layer
h_q0 = Dense(intermediate_dim0, activation='linear')(h_q1)  # ADDED intermediate layer
h_q = Dense(intermediate_dim, activation='linear')(h_q0)
mu = Dense(latent_dim, activation='linear')(h_q) # mean
log_sigma = Dense(latent_dim, activation='linear')(h_q) # log-sigma

# ----------------------------------------------------------------------------
# Reparametrization

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=epsilon_mean, stddev=epsilon_std)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# ----------------------------------------------------------------------------
# Decoder 

# P(X|z) -- decoder in 2 steps (for saving later)
decoder_hidden = Dense(latent_dim, activation='linear')
decoder_hidden0 = Dense(intermediate_dim, activation='linear') # ADDED intermediate layer
decoder_hidden1 = Dense(intermediate_dim0, activation='linear') # ADDED intermediate layer
decoder_hidden2 = Dense(intermediate_dim1, activation='linear') # ADDED intermediate layer
decoder_hidden3 = Dense(intermediate_dim2, activation='linear') # ADDED intermediate layer
decoder_hidden4 = Dense(intermediate_dim3, activation='linear') # ADDED intermediate layer
decoder_out = Dense(original_dim, activation='sigmoid')

h_p0 = decoder_hidden(z)
h_p1 = decoder_hidden0(h_p0)  # ADDED intermediate layer
h_p2 = decoder_hidden1(h_p1) # ADDED intermediate layer
h_p3 = decoder_hidden2(h_p2) # ADDED intermediate layer
h_p4 = decoder_hidden3(h_p3) # ADDED intermediate layer
h_p5 = decoder_hidden4(h_p4) # ADDED intermediate layer
outputs = decoder_out(h_p5)

# ----------------------------------------------------------------------------
# Saving models

# Whole VAE architecture
vae = Model(inputs, outputs)

# Encoder
encoder = Model(inputs, mu)
#encoder = Model(inputs, [mu, log_sigma])


# Decoder
decoder_input = Input(shape=(latent_dim,))

_h_decoded = decoder_hidden(decoder_input)
_h0_decoded = decoder_hidden0(_h_decoded)    # ADDED layer_1
_h1_decoded = decoder_hidden1(_h0_decoded)    # ADDED layer_1
_h2_decoded = decoder_hidden2(_h1_decoded)    # ADDED ---
_h3_decoded = decoder_hidden3(_h2_decoded)    # ADDED --- should replicate decoder arch
_h4_decoded = decoder_hidden4(_h3_decoded)  # ADDED --- should replicate decoder arch
_x_decoded_mean = decoder_out(_h4_decoded)  # mean of P(x*|z)
decoder = Model(decoder_input, _x_decoded_mean)

# ----------------------------------------------------------------------------
# Define loss

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)

    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5*K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl

# Define optimizer
vae.compile(optimizer='adam', loss=vae_loss)
K.set_value(vae.optimizer.lr, learning_rate)
K.set_value(vae.optimizer.decay, decay_rate)
print(vae.summary())

# Training
vae.fit(x_train, x_train, shuffle=True, batch_size=batch_size, nb_epoch=num_epochs, verbose=2, validation_data=(x_test, x_test))

# Saving
# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)
x_train_decoded = decoder.predict(x_train_encoded)

x_test_encoded = encoder.predict(x_test)
x_test_decoded = decoder.predict(x_test_encoded)

np.savetxt(DataDir+'encoded_xtrainP'+str(num_para)+'_'+ fileOut +'.txt', x_train_encoded)
np.savetxt(DataDir+'encoded_xtestP'+str(num_para)+'_'+ fileOut +'.txt', x_test_encoded)

# np.save(DataDir+'para5_'+str(num_train)+'.npy', y_train)
# -------------------- Save model/weights --------------------------


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    #val_loss = vae.history.history['val_loss']
    val_loss = np.ones_like(train_loss)  ## FAKE val loss -- since we are excluding this while training for now
    training_hist = np.vstack([epochs, train_loss, val_loss])

    vae.save(ModelDir+'fullAEP'+str(num_para)+'_' + fileOut + '.hdf5')
    encoder.save(ModelDir + 'EncoderP'+str(num_para)+'_' + fileOut + '.hdf5')
    decoder.save(ModelDir + 'DecoderP'+str(num_para)+'_' + fileOut + '.hdf5')
    np.savetxt(ModelDir + 'TrainingHistoryP'+str(num_para)+'_'+fileOut+'.txt', training_hist)

# -------------------- Plotting routines --------------------------
PlotScatter = True
if PlotScatter:
    # display a 2D plot of latent space (just 2 dimensions)
    plt.figure(figsize=(6, 6))

    x_train_encoded = encoder.predict(x_train)
    plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], c=y_train[:, 0], cmap='spring')
    plt.colorbar()

    x_test_encoded = encoder.predict(x_test)
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test[:, 0], cmap='copper')
    plt.colorbar()
    plt.title(fileOut)
    plt.savefig(PlotsDir + 'Scatter_z'+fileOut+'.png')

plotLoss = True
if plotLoss:
    import matplotlib.pylab as plt

    epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    #val_loss = vae.history.history['val_loss']

    val_loss = np.ones_like(train_loss)  ## FAKE val loss -- since we are excluding this while training for now

    fig, ax = plt.subplots(1,1, sharex= True, figsize = (8,6))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace= 0.02)
    ax.plot(epochs,train_loss, '-', lw =1.5)
    ax.plot(epochs,val_loss, '-', lw = 1.5)
    ax.set_ylabel('loss')
    ax.set_xlabel('epochs')
    # ax[0].set_ylim([0,1])
    # ax[0].set_title('Loss')
    ax.legend(['train loss','val loss'])
    plt.tight_layout()
    # plt.savefig(PlotsDir+'Training_loss.png')

PlotModel = False
if PlotModel:
    from keras.utils.vis_utils import plot_model
    fileOut = PlotsDir + 'ArchitectureFullAE.png'
    plot_model(vae, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = PlotsDir + 'ArchitectureEncoder.png'
    plot_model(encoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

    fileOut = PlotsDir + 'ArchitectureDecoder.png'
    plot_model(decoder, to_file=fileOut, show_shapes=True, show_layer_names=True)

plt.figure()
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(np.reshape(x_train[i], (33, 33)))
    # plt.title('Emulated image using PCA + GP '+str(i))
    # plt.colorbar()
    plt.subplot(2, 10, 10+i+1)
    plt.imshow(np.reshape(x_train_decoded[i], (33, 33)))
    # plt.title('Simulated image using GalSim '+str(i))
    # plt.colorbar()

plt.show()
