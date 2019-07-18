"""

Variational Autoencoding for CMB power spectra.

Training scheme is unsupervised, i.e., the autoencoder is not given information about
cosmological parameters.



This script uses Keras for neural network architecture.

"""
import numpy as np


from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras import losses

import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt
import keras.backend as K

import params_debug as params
# import Cl_load
# import SetPub
# SetPub.set_pub()


############### Setting same float, random seeds ##############

np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
K.set_floatx('float32')

###################### PARAMETERS ##############################

original_dim = params.original_dim # 2549
intermediate_dim3 = params.intermediate_dim3 # 1600
intermediate_dim2 = params.intermediate_dim2 # 1024
intermediate_dim1 = params.intermediate_dim1 # 512
intermediate_dim0 = params.intermediate_dim0 # 256
intermediate_dim = params.intermediate_dim # 256
latent_dim = params.latent_dim # 10

ClID = params.ClID
num_train = params.num_train # 512
num_test = params.num_test # 32
num_para = params.num_para # 5


batch_size = params.batch_size # 8
num_epochs = params.num_epochs # 100
epsilon_mean = params.epsilon_mean # 1.0
epsilon_std = params.epsilon_std # 1.0
learning_rate = params.learning_rate # 1e-3
decay_rate = params.decay_rate # 0.0

noise_factor = params.noise_factor # 0.00

######################## I/O ##################################

DataDir = params.DataDir
PlotsDir = params.PlotsDir
ModelDir = params.ModelDir

fileOut = params.fileOut


# ----------------------------- i/o ------------------------------------------

Trainfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_train)+'.txt')
Testfiles = np.loadtxt(DataDir + 'P'+str(num_para)+ClID+'Cl_'+str(num_test)+'.txt')

x_train = Trainfiles[:, num_para+2:]
x_test = Testfiles[:, num_para+2:]
y_train = Trainfiles[:, 0: num_para]
y_test =  Testfiles[:, 0: num_para]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

ls = np.loadtxt(DataDir+'P'+str(num_para)+'ls_'+str(num_train)+'.txt')[2:]

#------------------------- SCALAR parameter for rescaling -----------------------
#### ---- All the Cl's are rescaled uniformly #####################

'''
minVal = np.min( [np.min(x_train), np.min(x_test ) ])
meanFactor = 1.1*minVal if minVal < 0 else 0
# meanFactor = 0.0
print('-------mean factor:', meanFactor)
x_train = x_train - meanFactor #/ 255.
x_test = x_test - meanFactor #/ 255.

# x_train = np.log10(x_train) #x_train[:,2:] #
# x_test =  np.log10(x_test) #x_test[:,2:] #

normFactor = np.max( [np.max(x_train), np.max(x_test ) ])
# normFactor = 1
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.


np.savetxt(DataDir+'meanfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt', [meanFactor])
np.savetxt(DataDir+'normfactorP'+str(num_para)+ClID+'_'+ fileOut +'.txt', [normFactor])

'''
#########  New l-dependant rescaling (may not work with LSTMs) ###########
#### - --   works Ok with iid assumption ############


minVal = np.min( [np.min(x_train, axis = 0), np.min(x_test , axis = 0) ], axis=0)
#meanFactor = 1.1*minVal if minVal < 0 else 0
# meanFactor = 0.0
meanFactor = 1.1*minVal
print('-------mean factor:', meanFactor)
x_train = x_train - meanFactor #/ 255.
x_test = x_test - meanFactor #/ 255.

# x_train = np.log10(x_train) #x_train[:,2:] #
# x_test =  np.log10(x_test) #x_test[:,2:] #

normFactor = np.max( [np.max(x_train, axis = 0), np.max(x_test, axis = 0 ) ], axis = 0)
# normFactor = 1
print('-------normalization factor:', normFactor)
x_train = x_train.astype('float32')/normFactor #/ 255.
x_test = x_test.astype('float32')/normFactor #/ 255.


np.savetxt(DataDir+'meanfactorPArr'+str(num_para)+ClID+'_'+ fileOut +'.txt', [meanFactor])
np.savetxt(DataDir+'normfactorPArr'+str(num_para)+ClID+'_'+ fileOut +'.txt', [normFactor])


##############################################
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# # Trying to get x_train ~ (-1, 1) -- doesn't work well
# x_mean = np.mean(x_train, axis = 0)
# x_train = x_train - x_mean
# x_test = x_test - x_mean


## ADD noise
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# plt.plot(x_test_noisy.T, 'r', alpha = 0.3)
# plt.plot(x_test_noisy.T*(y_test[:,2]**2), 'b', alpha = 0.3)

x_train_noisy = K.cast_to_floatx(x_train_noisy)
x_train = K.cast_to_floatx(x_train)
# ------------------------------------------------------------------------------

################# ARCHITECTURE ###############################

# ----------------------------------------------------------------------------

# Q(z|X) -- encoder
inputs = Input(shape=(original_dim,))
h_q3 = Dense(intermediate_dim3, activation='linear')(inputs) # ADDED intermediate layer
h_q2 = Dense(intermediate_dim2, activation='linear')(h_q3) # ADDED intermediate layer
h_q1 = Dense(intermediate_dim1, activation='linear')(h_q2) # ADDED intermediate layer
h_q0 = Dense(intermediate_dim0, activation='linear')(h_q1) # ADDED intermediate layer
h_q = Dense(intermediate_dim, activation='linear')(h_q0)
mu = Dense(latent_dim, activation='linear')(h_q)
log_sigma = Dense(latent_dim, activation='linear')(h_q)

# ----------------------------------------------------------------------------

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(batch_size, latent_dim), mean=epsilon_mean, stddev=epsilon_std)
    return mu + K.exp(log_sigma / 2) * eps


# Sample z ~ Q(z|X)
z = Lambda(sample_z)([mu, log_sigma])

# ----------------------------------------------------------------------------

# P(X|z) -- decoder
decoder_hidden = Dense(latent_dim, activation='linear')
decoder_hidden0 = Dense(intermediate_dim, activation='linear') # ADDED intermediate layer
decoder_hidden1 = Dense(intermediate_dim0, activation='linear') # ADDED intermediate layer
decoder_hidden2 = Dense(intermediate_dim1, activation='linear') # ADDED intermediate layer
decoder_hidden3 = Dense(intermediate_dim2, activation='linear') # ADDED intermediate layer
decoder_hidden4 = Dense(intermediate_dim3, activation='linear') # ADDED intermediate layer
decoder_out = Dense(original_dim, activation='sigmoid')

h_p0 = decoder_hidden(z)
h_p1 = decoder_hidden0(h_p0) # ADDED intermediate layer
h_p2 = decoder_hidden1(h_p1) # ADDED intermediate layer
h_p3 = decoder_hidden2(h_p2) # ADDED intermediate layer
h_p4 = decoder_hidden3(h_p3) # ADDED intermediate layer
h_p5 = decoder_hidden4(h_p4) # ADDED intermediate layer
outputs = decoder_out(h_p5)

# ----------------------------------------------------------------------------


# Overall VAE model, for reconstruction and training
vae = Model(inputs, outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputs, mu)

# Generator model, generate new data given latent variable z
# d_in = Input(shape=(latent_dim,))
# d_h = decoder_hidden(d_in)
# d_h1 = decoder_hidden1(d_h)
# d_h2 = decoder_hidden2(d_h1)
# d_out = decoder_out(d_h2)
# decoder = Model(d_in, d_out)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))

_h_decoded = decoder_hidden(decoder_input)
_h0_decoded = decoder_hidden0(_h_decoded)    ## ADDED layer_1
_h1_decoded = decoder_hidden1(_h0_decoded)    ## ADDED layer_1
_h2_decoded = decoder_hidden2(_h1_decoded)    ## ADDED ---
_h3_decoded = decoder_hidden3(_h2_decoded)    ## ADDED --- should replicate decoder arch
_h4_decoded = decoder_hidden4(_h3_decoded)    ## ADDED --- should replicate decoder arch
_x_decoded_mean = decoder_out(_h4_decoded)
decoder = Model(decoder_input, _x_decoded_mean)


# -------------------------------------------------------------
#CUSTOM LOSS

def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # recon = K.categorical_crossentropy(y_pred, y_true)
    # recon = losses.mean_squared_error(y_pred, y_true)

    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5*K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

    return recon + kl



adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                          decay=decay_rate)

vae.compile(optimizer='adam', loss=vae_loss)

K.set_value(vae.optimizer.lr, learning_rate)
K.set_value(vae.optimizer.decay, decay_rate)


print(vae.summary())


# from keras.utils import plot_model
# plot_model(vae, show_shapes = True, show_layer_names = False,  to_file='model.png' )


#TRAIN


# checkpoint

Checkpoint = False

if Checkpoint:
    from keras.callbacks import ModelCheckpoint
    filepath= ModelDir+'CallbackfullAEP'+str(num_para)+ClID+'_' + fileOut + '.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    vae.fit(x_train_noisy, x_train, shuffle=True, batch_size=batch_size, nb_epoch=num_epochs, verbose=2,
            validation_data=(x_test_noisy, x_test), callbacks=callbacks_list)


else:
    #vae.fit(x_train_noisy, x_train, shuffle=True, batch_size=batch_size, nb_epoch=num_epochs, verbose=2, validation_data=(x_test_noisy, x_test))

    vae.fit(x_train_noisy, x_train, shuffle=True, batch_size=batch_size, nb_epoch=num_epochs, verbose=2) ## excluding validation for now -- otherwise creates problems for batch size > 8 


print('--------learning rate : ', K.eval(vae.optimizer.lr) )
# ----------------------------------------------------------------------------

x_train_encoded = encoder.predict(x_train)
x_train_decoded = decoder.predict(x_train_encoded)

x_test_encoded = encoder.predict(x_test)
x_test_decoded = decoder.predict(x_test_encoded)

np.savetxt(DataDir+'encoded_xtrainP'+str(num_para)+ClID+'_'+ fileOut +'.txt', x_train_encoded)
np.savetxt(DataDir+'encoded_xtestP'+str(num_para)+ClID+'_'+ fileOut +'.txt', x_test_encoded)

# np.save(DataDir+'para5_'+str(num_train)+'.npy', y_train)
# -------------------- Save model/weights --------------------------


SaveModel = True
if SaveModel:
    epochs = np.arange(1, num_epochs+1)
    train_loss = vae.history.history['loss']
    #val_loss = vae.history.history['val_loss']
    val_loss = np.ones_like(train_loss)  ## FAKE val loss -- since we are excluding this while training for now
    training_hist = np.vstack([epochs, train_loss, val_loss])


    vae.save(ModelDir+'fullAEP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    encoder.save(ModelDir + 'EncoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    decoder.save(ModelDir + 'DecoderP'+str(num_para)+ClID+'_' + fileOut + '.hdf5')
    np.savetxt(ModelDir + 'TrainingHistoryP'+str(num_para)+ClID+'_'+fileOut+'.txt', training_hist)

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
    plt.savefig( PlotsDir + 'Scatter_z'+ClID+fileOut+'.png')



PlotSample = True
if PlotSample:
    for i in range(10):
        plt.figure(91, figsize=(8,6))
        # plt.plot(ls, 10**(normFactor*x_train_decoded[i])/10**(normFactor*x_train[i]), 'r-', alpha = 0.8)
        # plt.plot(ls, 10**(normFactor*x_test_decoded[i])/10**(normFactor*x_test[i]), 'k-', alpha = 0.8)

        plt.plot(ls, x_train_decoded[i]/x_train[i], 'r-', alpha = 0.8)
        plt.plot(ls, x_test_decoded[i]/x_test[i], 'k-', alpha = 0.8)

        plt.ylim(0.85, 1.15)

        # plt.xscale('log')
        # plt.yscale('log')
        plt.ylabel('reconstructed/real')
        plt.title('train(red) and test (black)')
        plt.savefig(PlotsDir + 'Ratio_ttP'+str(num_para)+ClID+fileOut+'.png')


        if (i%2 == 1):
            plt.figure(654, figsize=(8,6))
            # plt.plot(ls, 10**(normFactor*x_test_decoded[i]), 'r-', alpha = 0.8)
            # plt.plot(ls, 10**(normFactor*x_test[i]), 'b--', alpha = 0.8)

            plt.plot(ls, ( normFactor*x_test_decoded[i] ) + meanFactor, 'r-', alpha = 0.8)
            plt.plot(ls, ( normFactor*x_test[i] ) + meanFactor, 'b--', alpha = 0.8)


            # plt.xscale('log')
            # plt.yscale('log')
            plt.title('Testing: reconstructed (red) and real (blue)')
            plt.savefig(PlotsDir + 'decoderTestP'+str(num_para)+ClID+ fileOut + '.png')

    plt.show()


print(fileOut)
print(ClID)
print('--------max ratio (train) : ', np.max(x_train_decoded/x_train) )
print('--------max ratio (test)  : ', np.max(x_test_decoded/x_test) )


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

plt.show()
