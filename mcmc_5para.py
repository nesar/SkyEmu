import numpy as np
import matplotlib.pylab as plt
# import corner
import emcee
import time
from keras.models import load_model
import params
import george
from george.kernels import Matern32Kernel


import pygtc


# import Cl_load
# import SetPub
# SetPub.set_pub()


#### parameters that define the MCMC

ndim = 5
nwalkers = 600  # 500
nrun_burn = 50  # 300
nrun = 300  # 700
fileID = 1


########## REAL DATA with ERRORS #############################
# Planck/SPT/WMAP data
# TE, EE, BB next

dirIn = '../Cl_data/RealData/'
allfiles = ['WMAP.txt', 'SPTpol.txt', 'PLANCKlegacy.txt']

lID = np.array([0, 2, 0])
ClID = np.array([1, 3, 1])
emaxID = np.array([2, 4, 2])
eminID = np.array([2, 4, 2])

print(allfiles)


# for fileID in [realDataID]:
with open(dirIn + allfiles[fileID]) as f:
    lines = (line for line in f if not line.startswith('#'))
    allCl = np.loadtxt(lines, skiprows=1)

    l = allCl[:, lID[fileID]].astype(int)
    Cl = allCl[:, ClID[fileID]]
    emax = allCl[:, emaxID[fileID]]
    emin = allCl[:, eminID[fileID]]

    print(l.shape)


############## GP FITTING ################################################################################
##########################################################################################################



def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


###################### PARAMETERS ##############################

original_dim = params.original_dim  # 2549
latent_dim = params.latent_dim  # 10

ClID = params.ClID
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

# ----------------------------- i/o ------------------------------------------



Trainfiles = np.loadtxt(DataDir + 'P' + str(num_para) + ClID + 'Cl_' + str(num_train) + '.txt')
Testfiles = np.loadtxt(DataDir + 'P' + str(num_para) + ClID + 'Cl_' + str(num_test) + '.txt')

x_train = Trainfiles[:, num_para + 2:]
x_test = Testfiles[:, num_para + 2:]
y_train = Trainfiles[:, 0: num_para]
y_test = Testfiles[:, 0: num_para]

print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
print(y_train.shape, 'train sequences')
print(y_test.shape, 'test sequences')

ls = np.loadtxt(DataDir + 'P' + str(num_para) + 'ls_' + str(num_train) + '.txt')[2:]

# ----------------------------------------------------------------------------

normFactor = np.loadtxt(DataDir + 'normfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')
meanFactor = np.loadtxt(DataDir + 'meanfactorP' + str(num_para) + ClID + '_' + fileOut + '.txt')

print('-------normalization factor:', normFactor)
print('-------rescaling factor:', meanFactor)

x_train = x_train - meanFactor  # / 255.
x_test = x_test - meanFactor  # / 255.

x_train = x_train.astype('float32') / normFactor  # / 255.
x_test = x_test.astype('float32') / normFactor  # / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# ------------------------------------------------------------------------------


################# ARCHITECTURE ###############################



LoadModel = True
if LoadModel:
    encoder = load_model(ModelDir + 'EncoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5')
    decoder = load_model(ModelDir + 'DecoderP' + str(num_para) + ClID + '_' + fileOut + '.hdf5')
    history = np.loadtxt(
        ModelDir + 'TrainingHistoryP' + str(num_para) + ClID + '_' + fileOut + '.txt')



##################################################33333

# Only required with george, not with GPy


kernel = Matern32Kernel([1000, 4000, 3000, 1000, 2000], ndim=num_para)

X1 = y_train[:, 0][:, np.newaxis]
X1a = rescale01(np.min(X1), np.max(X1), X1)

X2 = y_train[:, 1][:, np.newaxis]
X2a = rescale01(np.min(X2), np.max(X2), X2)

X3 = y_train[:, 2][:, np.newaxis]
X3a = rescale01(np.min(X3), np.max(X3), X3)

X4 = y_train[:, 3][:, np.newaxis]
X4a = rescale01(np.min(X4), np.max(X4), X4)

X5 = y_train[:, 4][:, np.newaxis]
X5a = rescale01(np.min(X5), np.max(X5), X5)

rescaledTrainParams = np.array(np.array([X1a, X2a, X3a, X4a, X5a])[:, :, 0])[:, np.newaxis]

# # ------------------------------------------------------------------------------
encoded_xtrain = np.loadtxt(
    DataDir + 'encoded_xtrainP' + str(num_para) + ClID + '_' + fileOut + '.txt').T
encoded_xtest_original = np.loadtxt(
    DataDir + 'encoded_xtestP' + str(num_para) + ClID + '_' + fileOut + '.txt')


# ------------------------------------------------------------------------------

def GPcompute(rescaledTrainParams, latent_dim):
    gp = {}
    for j in range(latent_dim):
        gp["fit{0}".format(j)] = george.GP(kernel)
        gp["fit{0}".format(j)].compute(rescaledTrainParams[:, 0, :].T)
    return gp


def GPfit(computedGP, para_array):
    para_array[0] = rescale01(np.min(X1), np.max(X1), para_array[0])
    para_array[1] = rescale01(np.min(X2), np.max(X2), para_array[1])
    para_array[2] = rescale01(np.min(X3), np.max(X3), para_array[2])
    para_array[3] = rescale01(np.min(X4), np.max(X4), para_array[3])
    para_array[4] = rescale01(np.min(X5), np.max(X5), para_array[4])

    test_pts = para_array[:num_para].reshape(num_para, -1).T

    # -------------- Predict latent space ----------------------------------------

    W_pred = np.array([np.zeros(shape=latent_dim)])
    W_pred_var = np.array([np.zeros(shape=latent_dim)])

    for j in range(latent_dim):
        W_pred[:, j], W_pred_var[:, j] = computedGP["fit{0}".format(j)].predict(encoded_xtrain[j],
                                                                                test_pts)

    # -------------- Decode from latent space --------------------------------------

    x_decoded = decoder.predict(W_pred)

    return (normFactor * x_decoded[0]) + meanFactor



### Using pre-trained GPy model #######################

import GPy


GPmodelOutfile = DataDir + 'GPy_model' + str(latent_dim) + ClID + fileOut
m1 = GPy.models.GPRegression.load_model(GPmodelOutfile + '.zip')


def GPyfit(GPmodelOutfile, para_array):


    test_pts = para_array.reshape(num_para, -1).T

    # -------------- Predict latent space ----------------------------------------

    # W_pred = np.array([np.zeros(shape=latent_dim)])
    # W_pred_var = np.array([np.zeros(shape=latent_dim)])

    m1p = m1.predict(test_pts)  # [0] is the mean and [1] the predictive
    W_pred = m1p[0]
    # W_varArray = m1p[1]


    # for j in range(latent_dim):
    #     W_pred[:, j], W_pred_var[:, j] = computedGP["fit{0}".format(j)].predict(encoded_xtrain[j],
    #                                                                             test_pts)

    # -------------- Decode from latent space --------------------------------------

    x_decoded = decoder.predict(W_pred.reshape(latent_dim, -1).T )

    return (normFactor * x_decoded[0]) + meanFactor


x_id = 20

x_decodedGPy = GPyfit(GPmodelOutfile, y_test[x_id])
# computedGP = GPcompute(rescaledTrainParams, latent_dim)
# x_decoded = GPfit(computedGP, y_test[x_id])

x_camb = (normFactor * x_test[x_id]) + meanFactor


plt.figure(1423)

# plt.plot(x_decoded, 'k--', alpha = 0.4, label = 'George')
plt.plot(x_decodedGPy, '--', alpha = 0.4 , label = 'GPy')
plt.plot(x_camb, 'r', alpha = 0.3 , label = 'camb')
plt.legend()
plt.show()


########################################################################################################################
########################################################################################################################

#### Cosmological Parameters ########################################

# OmegaM = np.linspace(0.12, 0.155, totalFiles)
# Omegab = np.linspace(0.0215, 0.0235, totalFiles)
# sigma8 = np.linspace(0.7, 0.9, totalFiles)
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)

#### Order of parameters: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s']
#        [label, true, min, max]

param1 = ["$\Omega_c h^2$", 0.1188, 0.12, 0.155] # Actual 0.119
param2 = ["$\Omega_b h^2$", 0.02230, 0.0215, 0.0235]
param3 = ["$\sigma_8$", 0.8159, 0.7, 0.9]
param4 = ["$h$", 0.6774, 0.55, 0.85]
param5 = ["$n_s$", 0.9667, 0.85, 1.05]

## Make sure the changes are made in log prior definition too. Variable: new_params


#
# OmegaM = np.linspace(0.10, 0.140, totalFiles)
# Omegab = np.linspace(0.0205, 0.0235, totalFiles)
# sigma8 = np.linspace(0.7, 0.9, totalFiles)
# h = np.linspace(0.55, 0.85, totalFiles)
# ns = np.linspace(0.85, 1.05, totalFiles)

param1 = ["$\Omega_c h^2$", 0.1188, 0.10, 0.14] # Actual 0.119
param2 = ["$\Omega_b h^2$", 0.02230, 0.0205, 0.0235]
param3 = ["$\sigma_8$", 0.8159, 0.7, 0.9]
param4 = ["$h$", 0.6774, 0.55, 0.85]
param5 = ["$n_s$", 0.9667, 0.85, 1.05]



#################### CHAIN INITIALIZATION ##########################

## 2 options

Uniform_init = True
if Uniform_init:
# Choice 1: chain uniformly distributed in the range of the parameters
    pos_min = np.array([param1[2], param2[2], param3[2], param4[2], param5[2]])
    pos_max = np.array([param1[3], param2[3], param3[3], param4[3], param5[3]])
    psize = pos_max - pos_min
    pos0 = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]



True_init = False
if True_init:
# Choice 2: chain is initialized in a tight ball around the expected values
    pos0 = [[param1[1]*1.2, param2[1]*0.8, param3[1]*0.9, param4[1]*1.1, param5[1]*1.2] +
           1e-3*np.random.randn(ndim) for i in range(nwalkers)]


MaxLikelihood_init = False
if MaxLikelihood_init:
# Choice 2b: Find expected values from max likelihood and use that for chain initialization
# Requires likehood function below to run first

    import scipy.optimize as op
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [param1[1], param2[1], param3[1], param4[1], param5[1]], args=(x, y, yerr))
    p1_ml, p2_ml, p3_ml, p4_ml, p5_ml = result["x"]
    print(result['x'])


    pos0 = [result['x']+1.e-4*np.random.randn(ndim) for i in range(nwalkers)]



# Visualize the initialization

PriorPlot = False

if PriorPlot:

    fig = corner.corner(pos0, labels=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        range=[[param1[2],param1[3]], [param2[2], param2[3]], [param3[2],param3[3]],
                        [param4[2],param4[3]], [param5[2],param5[3]]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]])
    fig.set_size_inches(10, 10)




######### MCMC #######################


x = l[l < ls.max()]
y = Cl[l < ls.max()]
yerr = emax[l < ls.max()]

## Sample implementation :
# http://eso-python.github.io/ESOPythonTutorials/ESOPythonDemoDay8_MCMC_with_emcee.html
# https://users.obs.carnegiescience.edu/cburns/ipynbs/Emcee.html

def lnprior(theta):
    p1, p2, p3, p4, p5 = theta
    # if 0.12 < p1 < 0.155 and 0.7 < p2 < 0.9:
    if param1[2] < p1 < param1[3] and param2[2] < p2 < param2[3] and param3[2] < p3 < param3[3] \
            and param4[2] < p4 < param4[3] and param5[2] < p5 < param5[3]:
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr):
    p1, p2, p3, p4, p5 = theta
    # new_params = np.array([p1, 0.0225, p2 , 0.74, 0.9])

    new_params = np.array([p1, p2, p3, p4, p5])
    # model = GPfit(computedGP, new_params)#  Using George -- with model training

    model = GPyfit(GPmodelOutfile, new_params)# Using GPy -- using trained model


    mask = np.in1d(ls, x)
    model_mask = model[mask]

    # return -0.5 * (np.sum(((y - model) / yerr) ** 2.))
    return -0.5 * (np.sum(((y - model_mask) / yerr) ** 2.))


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# Let us setup the emcee Ensemble Sampler
# It is very simple: just one, self-explanatory line

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

###### BURIN-IN #################

time0 = time.time()
# burnin phase
pos, prob, state = sampler.run_mcmc(pos0, nrun_burn)
sampler.reset()
time1 = time.time()
print('burn-in time:', time1 - time0)

###### MCMC ##################
time0 = time.time()
# perform MCMC
pos, prob, state = sampler.run_mcmc(pos, nrun)
time1 = time.time()
print('mcmc time:', time1 - time0)

samples = sampler.flatchain
samples.shape


###########################################################################
samples_plot = sampler.chain[:, :, :].reshape((-1, ndim))



np.savetxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
    nrun)  + ClID + '_'  + fileOut + allfiles[fileID][:-4] +'.txt', sampler.chain[:, :, :].reshape((-1, ndim)))

####### FINAL PARAMETER ESTIMATES #######################################


samples_plot  = np.loadtxt(DataDir + 'Sampler_mcmc_ndim' + str(ndim) + '_nwalk' + str(nwalkers) +
                         '_run' + str(nrun)  + ClID + '_' + fileOut + allfiles[fileID][:-4] +'.txt')

# samples = np.exp(samples)
p1_mcmc, p2_mcmc, p3_mcmc, p4_mcmc, p5_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                       zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print('mcmc results:', p1_mcmc[0], p2_mcmc[0], p3_mcmc[0], p4_mcmc[0], p5_mcmc[0])

####### CORNER PLOT ESTIMATES #######################################

CornerPlot = False
if CornerPlot:

    fig = corner.corner(samples_plot, labels=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        range=[[param1[2],param1[3]], [param2[2], param2[3]], [param3[2],param3[3]],
                        [param4[2],param4[3]], [param5[2],param5[3]]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        show_titles=True,  title_args={"fontsize": 10})


    fig.savefig(PlotsDir +'corner_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun) + ClID + '_'  + fileOut +allfiles[fileID][:-4] + '.pdf')


    fig = pygtc.plotGTC(samples_plot, paramNames=[param1[0], param2[0], param3[0], param4[0], param5[0]],
                        truths=[param1[1], param2[1], param3[1], param4[1], param5[1]],
                        figureSize='MNRAS_page')#, plotDensity = True, filledPlots = False,\smoothingKernel = 0, nContourLevels=3)


    fig.savefig(PlotsDir + 'pygtc_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun)  + ClID + '_' + fileOut + allfiles[fileID][:-4] +'.pdf')

####### FINAL PARAMETER ESTIMATES #######################################
#
# plt.figure(1432)
#
# xl = np.array([0, 10])
# for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
#     plt.plot(xl, m*xl+b, color="k", alpha=0.1)
# plt.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
# plt.errorbar(x, y, yerr=yerr, fmt=".k", alpha=0.1)



####### SAMPLER CONVERGENCE #######################################

ConvergePlot = False
if ConvergePlot:

    fig = plt.figure(13214)
    plt.xlabel('steps')
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)

    ax1.plot(np.arange(nrun), sampler.chain[:, :, 0].T, lw = 0.2, alpha = 0.9)
    ax1.text(0.9, 0.9, param1[0], horizontalalignment='center', verticalalignment='center',
             transform = ax1.transAxes, fontsize = 20)
    ax2.plot(np.arange(nrun), sampler.chain[:, :, 1].T, lw = 0.2, alpha = 0.9)
    ax2.text(0.9, 0.9, param2[0], horizontalalignment='center', verticalalignment='center',
             transform = ax2.transAxes, fontsize = 20)
    ax3.plot(np.arange(nrun), sampler.chain[:, :, 2].T, lw = 0.2, alpha = 0.9)
    ax3.text(0.9, 0.9, param3[0], horizontalalignment='center', verticalalignment='center',
             transform = ax3.transAxes, fontsize = 20)
    ax4.plot(np.arange(nrun), sampler.chain[:, :, 3].T, lw = 0.2, alpha = 0.9)
    ax4.text(0.9, 0.9, param4[0], horizontalalignment='center', verticalalignment='center',
             transform = ax4.transAxes, fontsize = 20)
    ax5.plot(np.arange(nrun), sampler.chain[:, :, 4].T, lw = 0.2, alpha = 0.9)
    ax5.text(0.9, 0.9, param5[0], horizontalalignment='center', verticalalignment='center',
             transform = ax5.transAxes, fontsize = 20)
    plt.show()

    fig.savefig(PlotsDir + 'convergence_' + str(ndim) + '_nwalk' + str(nwalkers) + '_run' + str(
        nrun)  + ClID + '_'  + fileOut + allfiles[fileID][:-4] +'.pdf')