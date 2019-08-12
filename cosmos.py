import galsim
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
import os
import h5py
import cmath
from astropy.io import fits


# Plotting routine
def plot_gal(dataset_real, dataset_param, psf):
    for i in range(10):
        plt.subplot(3, 10, i+1)
        plt.imshow(dataset_real[i])
        # plt.colorbar(shrink=0.5)
        plt.subplot(3, 10, 10+i+1)
        plt.imshow(dataset_param[i])

        plt.subplot(3, 10, 20+i+1)
        psfplot = np.real(np.fft.ifft2(psf[i]))
        plt.imshow(psfplot)
    plt.show()
    return psfplot


def load_params(index, catalog, params_data, params_labels):
    n_params = params_labels.shape[0]
    par_dic = catalog.getParametricRecord(index)
    params = np.zeros(n_params)
    ind = np.where(params_data[:]['IDENT'] == par_dic['IDENT'])[0][0]
    for i in range(n_params):
        params[i] = params_data[ind][params_labels[i]]
    # print(str(params[0])+'-----'+str(par_dic['mag_auto']))
    return params


def compute_translation(shape):
    K, L = shape
    trans = np.zeros(shape, dtype=np.complex)
    for k in range(K):
        for l in range(L):
            trans[k, l] = cmath.exp(-2*np.pi*k*(1/2)*1.j)*cmath.exp(-2*np.pi*l*(1/2)*1.j)
    return trans


def fft_shift_psf(psf, translation):
    return np.fft.fft2(psf)*translation


# Set dimensions
nx = 64
ny = 64
pixel_scale = 0.03
n_train = 65000
n_test = 16000

if not os.path.isdir('../Data/output_cosmos'):
    os.mkdir('../Data/output_cosmos')

# Set filenames
file_name_train = os.path.join('../Data/Cosmos/data', 'cosmos_real_trainingset_train_'+str(n_train)+'_test_'+str(n_test)+'.hdf5')
file_name_test = os.path.join('../Data/Cosmos/data', 'cosmos_real_testingset_train_'+str(n_train)+'_test_'+str(n_test)+'.hdf5')

# Set parameters labels
params_data = fits.getdata('../Data/Cosmos/data/lensing14.fits')
params_labels = np.array(['MAG_AUTO', 'FLUX_AUTO', 'FLUX_RADIUS', 'KEVIN_MSTAR', 'MNUV', 'MU', 'MB', 'MV', 'MG', 'MR', 'MI', 'MJ', 'MK', 'MNUV_MR', 'SFR_MED', 'SSFR_MED'])
n_params = params_labels.shape[0]


# Load catalogs
catalog_real = galsim.COSMOSCatalog()
catalog_param = galsim.COSMOSCatalog(use_real=False)

# --------- Initialize sets
# Training and testing real images
training_set = np.zeros((n_train, nx, ny))
testing_set = np.zeros((n_test, nx, ny))
# Training and testing parametric images
# training_parametric = np.zeros((n_train, nx, ny))
# testing_parametric = np.zeros((n_test, nx, ny))
# Training and testing parameters
training_params = np.zeros((n_train, n_params))
testing_params = np.zeros((n_test, n_params))
# Training and testing psfs
training_psf = np.zeros((n_train, nx, ny))
testing_psf = np.zeros((n_test, nx, ny))
# translation = compute_translation((nx, ny))

# Modify GSParams
big_fft_params = galsim.GSParams(maximum_fft_size=12300)


# Generating training set and params
print('Loading training set ...')
for ind in range(n_train):
    print(ind)
    gal_real = catalog_real.makeGalaxy(ind, noise_pad_size=nx * pixel_scale)
    gal_param = catalog_param.makeGalaxy(ind, noise_pad_size=ny * pixel_scale)
    psf = gal_real.original_psf
    # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
    final_real = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)
    # final_parametric = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)

    training_set[ind] = final_real.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    # training_parametric[ind] = final_parametric.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    training_params[ind] = load_params(ind, catalog_param, params_data, params_labels)
    training_psf[ind] = psf.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    # training_psf[ind] = fft_shift_psf(psf.drawImage(nx=nx, ny=ny, scale=pixel_scale).array, translation)

# Generating testing set and params
print('Loading testing set ...')
for ind in range(n_test):
    i = ind + n_train
    print(ind)
    gal_real = catalog_real.makeGalaxy(i, noise_pad_size=nx * pixel_scale)
    gal_param = catalog_param.makeGalaxy(i, noise_pad_size=ny * pixel_scale)
    psf = gal_real.original_psf
    # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
    final_real = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)
    # final_parametric = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)

    testing_set[ind] = final_real.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    # testing_parametric[ind] = final_parametric.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    testing_params[ind] = load_params(i, catalog_param, params_data, params_labels)
    testing_psf[ind] = psf.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    # testing_psf[ind] = fft_shift_psf(psf.drawImage(nx=nx, ny=ny, scale=pixel_scale).array, translation)

mean = np.mean(training_params, axis=0)
mult = np.max(training_params - mean, axis=0)
y_train = (training_params - mean) * mult**-1
y_test = (testing_params - mean) * mult**-1

f, a = plt.subplots(4, 4, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 4})

plot = True
if plot:
    for i in range(4):
        for j in range(i+1):
            if(i != j):
                a[i, j].scatter(y_train[:, i], y_train[:, j], s=1, alpha=0.7)
                a[i, j].scatter(y_test[:, i], y_test[:, j], s=1, alpha=0.7)
                a[i, j].grid(True)
                a[j, i].set_visible(False)
                # plt.xticks([])
                # plt.yticks([])
            else:
                # a[i, i].text(0.4, 0.4, params[i], size='x-large')
                hist, bin_edges = np.histogram(y_train[:, i], density=True, bins=12)
                a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
                # plt.xticks([])
                # plt.yticks([])
    plt.show()

# Saving all datasets and params
print('Saving data sets ...')
f = h5py.File(file_name_train, 'w')
f.create_dataset('real galaxies', data=training_set)
# f.create_dataset('parametric galaxies', data=training_parametric)
f.create_dataset('parameters', data=training_params)
f.create_dataset('psf', data=training_psf)
f.close()

f = h5py.File(file_name_test, 'w')
f.create_dataset('real galaxies', data=testing_set)
# f.create_dataset('parametric galaxies', data=testing_parametric)
f.create_dataset('parameters', data=testing_params)
f.create_dataset('psf', data=testing_psf)
f.close()
