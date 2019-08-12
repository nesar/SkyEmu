import numpy as np
import GPy
import h5py
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import load_model, Model
import tensorflow as tf
import  tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
import tensorflow_hub as hub
from  flow import masked_autoregressive_conditional_template


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


n_cond = 16


def make_flow_fn(latent_size, maf_layers, maf_size, shift_only, activation):
    """ Creates a flow function with provided parameters
    """
    def flow_fn(cond, is_training):
        def init_once(x, name, trainable=False):
            return tf.get_variable(name, initializer=x, trainable=trainable)

        # Apply batch normalization on the inputs
        cond = tf.layers.batch_normalization(cond, axis=-1, training=is_training)

        chain = []
        for i in range(maf_layers):
            if i < 2:
                chain.append(tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=masked_autoregressive_conditional_template(
                                hidden_layers=[maf_size, maf_size],
                                conditional_tensor=cond,
                                activation=activation,
                                shift_only=False, name='maf%d'%i)))
            else:
                chain.append(tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=masked_autoregressive_conditional_template(
                                hidden_layers=[maf_size, maf_size],
                                conditional_tensor=cond,
                                activation=activation,
                                shift_only=shift_only, name='maf%d'%i)))
            chain.append(tfb.Permute(permutation=init_once(
                                 np.random.permutation(latent_size).astype("int32"),
                                 name='permutation%d'%i)))
        chain = tfb.Chain(chain)

        flow = tfd.TransformedDistribution(
                distribution=tfd.MultivariateNormalDiag(loc=np.zeros(latent_size, dtype='float32'),
                                                        scale_diag=init_once(np.ones(latent_size, dtype='float32'),
                                                        name='latent_scale', trainable=(not shift_only))),
                bijector=chain)

        return flow

    return flow_fn


def flow_model_fn(features, labels, mode, params, config):
    """
    Model function to create a VAE estimator
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        y = features
        def flow_module_spec():
            cond_layer = tf.placeholder(tf.float32, shape=[None, n_cond])
            flow = params['flow_fn'](cond_layer, is_training)
            hub.add_signature(inputs=cond_layer,
                              outputs=flow.sample(tf.shape(cond_layer)[0]))

        flow_spec = hub.create_module_spec(flow_module_spec)
        flow = hub.Module(flow_spec, name='flow_module')
        hub.register_module_for_export(flow, "code_sampler")
        predictions = {'code': flow(y)}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    x = features['x']
    y = features['y']

    # Loads the encoding function to work on the images
    code = x

    with tf.variable_scope("flow_module"):
        cond_layer = y
        flow = params['flow_fn'](cond_layer, is_training)
        loglikelihood = flow.log_prob(code)

    # This is the loglikelihood of a batch of images
    tf.summary.scalar('loglikelihood', tf.reduce_mean(loglikelihood))
    loss = - tf.reduce_mean(loglikelihood)

    # Training of the model
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"],
                                          global_step,
                                          params["max_steps"])

    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)

    eval_metric_ops = {"loglikelihood": tf.metrics.mean(tf.reduce_mean(loglikelihood))}

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)

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

print('Flow training ...')
params = { 'flow_fn': make_flow_fn(latent_size=32,
                                     maf_layers=4,
                                     maf_size=256,
                                     shift_only=True,
                                     activation=tf.nn.leaky_relu)}
model_dir='model_dir/flow'
batch_size=32
# Build estimator
estimator = tf.estimator.Estimator(
      flow_model_fn,
      params=params,
      config=tf.estimator.RunConfig(model_dir=model_dir))

def input_fn_train():
    code = tf.data.Dataset.from_tensor_slices(x_train_encoded)
    cond = tf.data.Dataset.from_tensor_slices(y_train)
    dset = tf.data.Dataset.zip((code, cond))
    dset = dset.repeat().shuffle(buffer_size=1000).batch(batch_size).prefetch(16)
    iterator = dset.make_one_shot_iterator()
    batch_code, batch_cond = iterator.get_next()
    return {'x': batch_code, 'y': batch_cond}

estimator.train(input_fn=input_fn_train, max_steps=30000)
print('GP pediction ...')

def input_fn_test():
    dset = tf.data.Dataset.from_tensor_slices(y_test)
    iterator = dset.make_one_shot_iterator()
    batch_cond = iterator.get_next()
    return batch_cond

pred = estimator.predict(input_fn_test, yield_single_examples=True)
x_test_gp_encoded = np.array([p for p in pred])
np.savetxt(DataDir + 'models/cvae_cosmos_gp_encoded_xtest_'+str(n_train)+'_'+str(n_test)+'.txt', x_test_gp_encoded)

print('Decoding ...')
x_test_gp_decoded = decoder2.predict([decoder1.predict(x_test_gp_encoded), psf_test])
np.savetxt(DataDir + 'models/cvae_cosmos_gp_decoded_xtest_'+str(n_train)+'_'+str(n_test)+'.txt', np.reshape(x_test_gp_decoded, (n_test, nx*ny)))
