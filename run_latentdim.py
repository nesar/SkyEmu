import numpy as np
import gengal
import pcagp_emulator as pcagp
import gp_model
import cvae
import testing
import matplotlib.pyplot as plt
import latinHyp_plot as lhc


DataDir = '../Data/'
task = 'latent_dim'

# Training and testing sets parameters
n_train = 1072
n_test = 100
n_params = 5

# Generate latin hypercubes
filename_train_par = 'lhc_'+str(n_train)+'_'+str(n_params)+'_training.txt'
lhc.latin_hc(filename_train_par, n_train)
filename_test_par = 'lhc_'+str(n_test)+'_'+str(n_params)+'_testing.txt'
lhc.latin_hc(filename_test_par, n_test)

# Generate training and testing sets
filename_train_gal = 'plot_'+str(n_train)+'_'+str(n_params)+'_training'
gengal.SaveGal(gengal.GenSetGal(DataDir+filename_train_par), filename_train_gal, 'galaxies')

filename_test_gal = 'plot_'+str(n_test)+'_'+str(n_params)+'_testing'
gengal.SaveGal(gengal.GenSetGal(DataDir+filename_test_par), filename_test_gal, 'galaxies')

# Latent space / truncated basis parameters
n_lat = 50
latent_dim = np.arange(n_lat)+1

# Initialize mse vectors
mse_pcagp = np.zeros(n_lat+1)
mse_vaegp = np.zeros(n_lat+1)

for l in latent_dim:
    # Compute mse from pca+gp emulator
    mse_pcagp[l-1] = pcagp.perform_pca_gp(l, task, filename_train_gal, filename_train_par, filename_test_gal, filename_test_par)

    # Train/Run cvae emulator and compute mse
    filename_decoder, filename_training_encoded = cvae.train_cvae(l, task, filename_train_gal, filename_test_gal)
    filename_test_encoded = gp_model.gp(task, filename_train_par, filename_test_par, filename_training_encoded)
    mse_vaegp[l-1] = testing.compute_vae_mse(filename_decoder, filename_test_encoded, filename_train_gal, filename_test_gal)

# Save mse vectors
np.savetxt(DataDir + 'MSE/' + 'mse_pcagp_latent_dim_' + str(n_lat) + '.txt', mse_pcagp)
np.savetxt(DataDir + 'MSE/' + 'mse_vaegp_latent_dim_' + str(n_lat) + '.txt', mse_vaegp)

# Plotting routine
plt.plot(latent_dim, mse_pcagp, label='PCA+GP emulator')
plt.plot(latent_dim, mse_vaegp, label='VAE+GP emulator')
plt.xlabel('Latent space / Truncated basis dimension')
plt.ylabel('Median MSE')
plt.legend()
plt.show()
