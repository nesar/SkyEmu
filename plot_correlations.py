import numpy as np
import matplotlib.pyplot as plt


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


# Load parameters and latent space for testing set
x_test_params = np.loadtxt('../Data/lhc_64_5_testing.txt')
x_test_encoded = np.loadtxt('../Data/x_test_encoded_64_5.txt')*10

# Load parameters and latent space for testing set
x_train_params = np.loadtxt('../Data/lhc_512_5.txt')
x_train_encoded = np.loadtxt('../Data/cvae_encoded_xtrainP.txt')*10

params_dim = x_train_params.shape[1]
latent_dim = x_train_encoded.shape[1]

# Rescale params
par_min = np.min(x_train_params, axis=0)
par_max = np.max(x_train_params - par_min, axis=0)
x_train_params = (x_train_params - par_min) * par_max**-1
x_test_params = (x_test_params - par_min) * par_max**-1

enc_min = np.min(x_train_encoded, axis=0)
enc_max = np.max(x_train_encoded - enc_min, axis=0)
x_train_encoded = (x_train_encoded - enc_min) * enc_max**-1
x_test_encoded = (x_test_encoded - enc_min) * enc_max**-1
# x_train_encoded, mean, mult = rescale(x_train_encoded)
# x_test_encoded = (x_test_encoded - mean) * mult**-1

n = latent_dim//2

# # for j in range(params_dim):
# #     plt.figure(j+1)
# #     for i in range(latent_dim):
# #         plt.subplot(2, n, i+1)
# #         plt.scatter(x_train_params[:, j], x_train_encoded[:, i], c='coral', s=1)
# #         plt.scatter(x_test_params[:, j], x_test_encoded[:, i], c='k', s=1)
# #         plt.xlabel('Param. '+str(j+1))
# #         plt.ylabel('Latent var. '+str(i+1))
# #         plt.xticks([])
# #         plt.yticks([])

# # plt.show()

params = ['Flux', 'Radius', 'Shear profile g1', 'Shear profile g2', 'PSF fwhm']

# for i in range(latent_dim):
#     plt.figure(i+1, (8, 3))
#     for j in range(params_dim):
#         plt.subplot(1, params_dim, j+1)
#         plt.scatter(x_train_params[:, j], x_train_encoded[:, i], c='coral', s=1)
#         plt.scatter(x_test_params[:, j], x_test_encoded[:, i], c='k', s=1)
#         plt.xlabel(params[j])
#         plt.ylabel('Latent var. '+str(i+1))
#         plt.xticks([])
#         plt.yticks([])
#     plt.savefig('../Plots/Correlation_plots/cvae_512_64_5par_20lat_1000epochs_latent'+str(i+1)+'_vs_params.pdf')

# plt.show()

total_dim = params_dim + latent_dim

f, a = plt.subplots(total_dim, total_dim, sharex=True, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.rcParams.update({'font.size': 4})

for i in range(params_dim):
    for j in range(i+1):
        if(i != j):
            a[i, j].scatter(x_train_params[:, i], x_train_params[:, j], s=1, alpha=0.7)
            a[i, j].scatter(x_test_params[:, i], x_test_params[:, j], s=1, alpha=0.7)
            a[i, j].grid(True)
            a[j, i].set_visible(False)
            plt.xticks([])
            plt.yticks([])
        else:
            # a[i, i].text(0.4, 0.4, params[i], size='x-large')
            hist, bin_edges = np.histogram(x_train_params[:, i], density=True, bins=12)
            a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
            plt.xticks([])
            plt.yticks([])

for i in range(latent_dim):
    for j in range(params_dim):
        a[i+params_dim, j].scatter(x_train_encoded[:, i], x_train_params[:, j], s=1, alpha=0.7)
        a[i+params_dim, j].scatter(x_test_encoded[:, i], x_test_params[:, j], s=1, alpha=0.7)
        a[i+params_dim, j].grid(True)
        a[j, i+params_dim].set_visible(False)
        plt.xticks([])
        plt.yticks([])

for i in range(latent_dim):
    for j in range(i+1):
        if(i != j):
            a[i+params_dim, j+params_dim].scatter(x_train_encoded[:, i], x_train_encoded[:, j], s=1, alpha=0.7)
            a[i+params_dim, j+params_dim].scatter(x_test_encoded[:, i], x_test_encoded[:, j], s=1, alpha=0.7)
            a[i+params_dim, j+params_dim].grid(True)
            a[j+params_dim, i+params_dim].set_visible(False)
            plt.xticks([])
            plt.yticks([])
        else:
            # a[i+params_dim, i+params_dim].text(0.4, 0.4, 'Latent space '+str(i), size='x-large')
            hist, bin_edges = np.histogram(x_train_encoded[:, i], density=True, bins=12)
            a[i+params_dim, i+params_dim].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
            plt.xticks([])
            plt.yticks([])

plt.tight_layout()
plt.savefig('../Data/Plots/Latent_vs_params_full_corr.pdf', figsize=(20000, 20000), bbox_inches="tight")
plt.show()
