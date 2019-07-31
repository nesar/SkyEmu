import galsim
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


# Plotting routine
def plot_gal(dataset_real, dataset_param):
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(dataset_real[i])
        # plt.colorbar(shrink=0.5)
        plt.subplot(2, 10, 10+i+1)
        plt.imshow(dataset_param[i])
    plt.show()


def load_params(index, catalog, n_params):
    par_dic = catalog.getParametricRecord(index)
    params = np.zeros(n_params)
    if par_dic['use_bulgefit']:
        # Fill bulge profile params
        params[4] = par_dic['flux'][1]
        params[5] = par_dic['hlr'][1]
        params[6] = par_dic['bulgefit'][11]
        params[7] = par_dic['bulgefit'][15]
        # Fill disk profile params
        params[8] = par_dic['flux'][2]
        params[9] = par_dic['hlr'][2]
        params[10] = par_dic['bulgefit'][3]
        params[11] = par_dic['bulgefit'][7]
    elif par_dic['viable_sersic']:
        # Fill sersic profiles params
        params[0] = par_dic['flux'][0]
        params[1] = par_dic['hlr'][0]
        params[2] = par_dic['sersicfit'][3]
        params[3] = par_dic['sersicfit'][7]
    return params


# Set dimensions
nx = 64
ny = 64
pixel_scale = 0.04
n_train = 2048
n_test = 128

if not os.path.isdir('../Data/output_cosmos'):
    os.mkdir('../Data/output_cosmos')

# Set filenames
file_name_train = os.path.join('../Data/output_cosmos', 'cosmos_real_train_'+str(n_train)+'.hdf5')
file_name_test = os.path.join('../Data/output_cosmos', 'cosmos_real_test_'+str(n_test)+'.hdf5')

# Set parameters labels
params_labels = np.array(['flux_sersic', 'hlr_sersic', 'q_sersic', 'phi_sersic', 'flux_bulge', 'hlr_bulge', 'q_bulge', 'phi_bulge', 'flux_disk', 'hlr_disk', 'q_disk', 'phi_disk'])
n_params = params_labels.shape[0]

# Load catalogs
catalog_real = galsim.COSMOSCatalog()
catalog_param = galsim.COSMOSCatalog(use_real=False)

# --------- Initialize sets
# Training and testing real images
training_set = np.zeros((n_train, nx, ny))
testing_set = np.zeros((n_test, nx, ny))
# Training and testing parametric images
training_parametric = np.zeros((n_train, nx, ny))
testing_parametric = np.zeros((n_test, nx, ny))
# Training and testing parameters
training_params = np.zeros((n_train, n_params))
testing_params = np.zeros((n_test, n_params))

# Modify GSParams
big_fft_params = galsim.GSParams(maximum_fft_size=12300)


# Generating training set and params
print('Loading training set ...')
for ind in range(n_train):
    gal_real = catalog_real.makeGalaxy(ind, noise_pad_size=nx * pixel_scale)
    gal_param = catalog_param.makeGalaxy(ind, noise_pad_size=ny * pixel_scale)
    psf = gal_real.original_psf
    # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
    final_real = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)
    final_parametric = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)

    training_set[ind] = final_real.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    training_parametric[ind] = final_parametric.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    training_params[ind] = load_params(ind, catalog_param, n_params)

# Generating testing set and params
print('Loading testing set ...')
for ind in range(n_test):
    i = ind + n_train
    gal_real = catalog_real.makeGalaxy(i, noise_pad_size=nx * pixel_scale)
    gal_param = catalog_param.makeGalaxy(i, noise_pad_size=ny * pixel_scale)
    psf = gal_real.original_psf
    # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
    final_real = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)
    final_parametric = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)

    testing_set[ind] = final_real.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    training_parametric[ind] = final_parametric.drawImage(nx=nx, ny=ny, scale=pixel_scale).array
    testing_params[ind] = load_params(ind, catalog_param, n_params)

# Saving all datasets and params
print('Saving data sets ...')
f = h5py.File(file_name_train, 'w')
f.create_dataset('real galaxies', data=training_set)
f.create_dataset('parametric galaxies', data=training_parametric)
f.create_dataset('parameters', data=training_params)
f.close()

f = h5py.File(file_name_test, 'w')
f.create_dataset('real galaxies', data=testing_set)
f.create_dataset('parametric galaxies', data=training_parametric)
f.create_dataset('parameters', data=testing_params)
f.close()
