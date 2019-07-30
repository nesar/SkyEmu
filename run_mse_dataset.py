import numpy as np
import gengal
import pcagp_emulator as pcagp
import gp_model
import cvae
import testing
import matplotlib.pyplot as plt


num_lhc = 100
num_evals = np.logspace(2, 4, num=num_lhc)
num_params = 5

latent_dim_vae = 20
latent_dim_pca = 12

num_test = 100
DataDir = '../Data/'
task = 'dataset_size'
filename_test_par = 'lhc_'+str(num_test)+'_'+str(num_params)+'_testing.txt'
# Generate the testing set
filename_test_gal = 'plot_'+str(num_test)+'_'+str(num_params)+'_testing'
gengal.SaveGal(gengal.GenSetGal(DataDir+filename_test_par), filename_test_gal, 'galaxies')

mse_pcagp = np.zeros(num_lhc)
mse_vaegp = np.zeros(num_lhc)

for i in range(num_lhc):
    nevals = np.int(num_evals[i])
    filename_training_par = 'lhc_'+str(nevals)+'_'+str(num_params)+'_training.txt'
    filename_training_gal = 'plot_'+str(nevals)+'_'+str(num_params)+'_training'
    # Generate the training set with latin hypercube parameters
    gengal.SaveGal(gengal.GenSetGal(DataDir+filename_training_par), filename_training_gal, 'galaxies')
    # Run pca+gp emulator and compute mse
    mse_pcagp[i] = pcagp.perform_pca_gp(latent_dim_pca, task, filename_training_gal, DataDir+filename_training_par, filename_test_gal, DataDir+filename_test_par)

    # Train/Run cvae emulator and compute mse
    filename_decoder, filename_training_encoded = cvae.train_cvae(latent_dim_vae, task, filename_training_gal, filename_test_gal)
    filename_test_encoded = gp_model.gp(task, filename_training_par, filename_test_par, filename_training_encoded)
    mse_vaegp[i] = testing.compute_vae_mse(filename_decoder, filename_test_encoded, filename_training_gal, filename_test_gal)

# Save mse vectors
np.savetxt(DataDir + 'MSE/' + 'mse_pcagp_training_set_size_' + str(num_lhc) + '.txt', mse_pcagp)
np.savetxt(DataDir + 'MSE/' + 'mse_vaegp_training_set_size_' + str(num_lhc) + '.txt', mse_vaegp)

# Plotting routine
plt.plot(num_evals, mse_pcagp, label='PCA+GP emulator')
plt.plot(num_evals, mse_vaegp, label='VAE+GP emulator')
plt.xlabel('Training set size')
plt.ylabel('Median MSE')
plt.legend()
plt.show()
