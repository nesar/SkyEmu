import numpy as np
import matplotlib.pyplot as plt

# Load parameters and latent space for testing set
x_test_params = np.loadtxt('../Data/lhc_64_5_testing.txt')
x_test_encoded = np.loadtxt('../Data/x_test_encoded_64_5.txt')

# Load parameters and latent space for testing set
x_train_params = np.loadtxt('../Data/lhc_512_5.txt')
x_train_encoded = np.loadtxt('../Data/cvae_encoded_xtrainP.txt')

params_dim = x_train_params.shape[1]
latent_dim = x_train_encoded.shape[1]

n = latent_dim//2

# for j in range(params_dim):
#     plt.figure(j+1)
#     for i in range(latent_dim):
#         plt.subplot(2, n, i+1)
#         plt.scatter(x_train_params[:, j], x_train_encoded[:, i], c='coral', s=1)
#         plt.scatter(x_test_params[:, j], x_test_encoded[:, i], c='k', s=1)
#         plt.xlabel('Param. '+str(j+1))
#         plt.ylabel('Latent var. '+str(i+1))
#         plt.xticks([])
#         plt.yticks([])

# plt.show()

params = ['Flux', 'Radius', 'Shear profile g1', 'Shear profile g2', 'PSF fwhm']

for i in range(latent_dim):
    plt.figure(i+1, (8, 3))
    for j in range(params_dim):
        plt.subplot(1, params_dim, j+1)
        plt.scatter(x_train_params[:, j], x_train_encoded[:, i], c='coral', s=1)
        plt.scatter(x_test_params[:, j], x_test_encoded[:, i], c='k', s=1)
        plt.xlabel(params[j])
        plt.ylabel('Latent var. '+str(i+1))
        plt.xticks([])
        plt.yticks([])
    plt.savefig('../Plots/Correlation_plots/cvae_512_64_5par_20lat_1000epochs_latent'+str(i+1)+'_vs_params.pdf')

# plt.show()
