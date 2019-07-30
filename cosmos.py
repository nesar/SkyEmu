import galsim
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


def plot_gal(dataset):
    for i in range(20):
        plt.subplot(2, 10, i+1)
        plt.imshow(dataset[i])
        plt.colorbar()
    plt.show()


nx = 64
ny = 64
pixel_scale = 0.04
n_train = 512
n_test = 64

if not os.path.isdir('../Data/output_cosmos'):
    os.mkdir('../Data/output_cosmos')

file_name_train = os.path.join('../Data/output_cosmos', 'cosmos_real_train_'+str(n_train)+'.hdf5')
file_name_test = os.path.join('../Data/output_cosmos', 'cosmos_real_test_'+str(n_test)+'.hdf5')

catalog_real = galsim.COSMOSCatalog()
catalog_param = galsim.COSMOSCatalog(use_real=False)

training_set = np.zeros((n_train, nx, ny))
testing_set = np.zeros((n_test, nx, ny))

big_fft_params = galsim.GSParams(maximum_fft_size=12300)

print('Loading training set ...')
for ind in range(n_train):
    gal_real = catalog_real.makeGalaxy(ind, noise_pad_size=nx * pixel_scale)
    gal_param = catalog_param.makeGalaxy(ind, noise_pad_size=ny * pixel_scale)
    psf = gal_real.original_psf
    # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
    final = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)

    training_set[ind] = final.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

print('Loading testing set ...')
for ind in range(n_test):
    i = ind + n_train
    gal_real = catalog_real.makeGalaxy(i, noise_pad_size=nx * pixel_scale)
    gal_param = catalog_param.makeGalaxy(i, noise_pad_size=ny * pixel_scale)
    psf = gal_real.original_psf
    # final = galsim.Convolve([gal_param, psf], gsparams=big_fft_params)
    final = galsim.Convolve([gal_real, psf], gsparams=big_fft_params)

    testing_set[ind] = final.drawImage(nx=nx, ny=ny, scale=pixel_scale).array

print('Saving data sets ...')
f = h5py.File(file_name_train, 'w')
f.create_dataset('galaxies', data=training_set)
f.close()

f = h5py.File(file_name_test, 'w')
f.create_dataset('galaxies', data=testing_set)
f.close()
