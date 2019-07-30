import galsim
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


# Plotting routine
def plot_gal(dataset):
    for i in range(20):
        plt.subplot(2, 10, i+1)
        plt.imshow(dataset[i])
        plt.colorbar()
    plt.show()


# Set dimensions
nx = 64
ny = 64
pixel_scale = 0.04
n_train = 1025
n_test = 64

if not os.path.isdir('../Data/output_cosmos'):
    os.mkdir('../Data/output_cosmos')

# Set filenames
file_name_train = os.path.join('../Data/output_cosmos', 'cosmos_real_train_'+str(n_train)+'.hdf5')
file_name_test = os.path.join('../Data/output_cosmos', 'cosmos_real_test_'+str(n_test)+'.hdf5')
file_name_train_params = os.path.join('../Data/output_cosmos', 'cosmos_real_train_'+str(n_train)+'_params.hdf5')
file_name_test_params = os.path.join('../Data/output_cosmos', 'cosmos_real_test_'+str(n_test)+'_params.hdf5')
file_name_labels_params = os.path.join('../Data/output_cosmos', 'cosmos_real_labels_params.hdf5')

# Set parameters records
params_labels = np.array(['flux_sersic', 'hlr_sersic', 'q_sersic', 'phi_sersic', 'flux_bulge', 'hlr_bulge', 'q_bulge', 'phi_bulge', 'flux_disk', 'hlr_disk', 'q_disk', 'phi_disk'])
n_params = params_labels.shape[0]

# Load catalogs
catalog_real = galsim.COSMOSCatalog()
catalog_param = galsim.COSMOSCatalog(use_real=False)

# Initialize sets
training_set = np.zeros((n_train, nx, ny))
testing_set = np.zeros((n_test, nx, ny))
training_params = np.zeros((n_train, n_params))
testing_params = np.zeros((n_test, n_params))

# Modify GSParams
big_fft_params = galsim.GSParams(maximum_fft_size=12300)

param_dict = catalog_param.getParametricRecord(0)
print(param_dict)


# # Generating training set and params
# print('Loading training set ...')
# for ind in range(n_train):
#     gal_real = catalog_real.makeGalaxy(ind, noise_pad_size=nx * pixel_scale)
#     gal_param = catalog_param.makeGalaxy(ind, noise_pad_size=ny * pixel_scale)
#     psf = gal_real.original_psf
#     # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
#     final = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)

#     training_set[ind] = final.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

# # Generating testing set and params
# print('Loading testing set ...')
# for ind in range(n_test):
#     i = ind + n_train
#     gal_real = catalog_real.makeGalaxy(i, noise_pad_size=nx * pixel_scale)
#     gal_param = catalog_param.makeGalaxy(i, noise_pad_size=ny * pixel_scale)
#     psf = gal_real.original_psf
#     # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
#     final = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)

#     testing_set[ind] = final.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

# # Saving all datasets and params
# print('Saving data sets ...')
# f = h5py.File(file_name_train, 'w')
# f.create_dataset('galaxies', data=training_set)
# f.close()

# f = h5py.File(file_name_test, 'w')
# f.create_dataset('galaxies', data=testing_set)
# f.close()

# f = h5py.File(file_name_train_params, 'w')
# f.create_dataset('galaxies', data=training_params)
# f.close()

# f = h5py.File(file_name_test_params, 'w')
# f.create_dataset('galaxies', data=testing_params)
# f.close()

# f = h5py.File(file_name_labels_params, 'w')
# f.create_dataset('galaxies', data=params_labels)
# f.close()