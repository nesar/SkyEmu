import numpy as np
import matplotlib.pyplot as plt
import galsim

# ------------------------- Computing assessments ---------------------------


def mse_r2(true, predicted):
    """
    Compute the mean square error (mse) and the r squared error (r2) of the predicted set of images.
    Inputs :
    - true : the original simulated set of images (n_imgs, nx, ny)
    - predicted : reconstructed set of images (n_imgs, nx, ny)
    Outputs :
    - mse : Mean Square Error
    - r2 : R-squared Error
    """
    # Reshaping set of images
    n_imgs, nx, ny = true.shape
    true = np.reshape(true, (n_imgs, nx*ny))
    predicted = np.reshape(predicted, (n_imgs, nx*ny))

    # Compute MSE
    se = np.sum((true - predicted)**2, axis=1)
    mse = se*(nx*ny)**-1

    # Compute R squared
    mean = np.mean(true, axis=1)
    r2 = 1 - se*np.sum((true - np.expand_dims(mean, axis=1))**2, axis=1)**-1

    plot = True
    if plot:
        plt.figure('MSE distribution')
        h = plt.hist(mse, bins=50, density=True)
        plt.xlabel('MSE')
        plt.ylabel('Density')
        plt.title('MSE distribution (min = '+str(np.min(mse))+', max = '+str(np.max(mse))+', median = '+str(np.median(mse))+').')

        plt.figure('R$^2$ distribution')
        h = plt.hist(r2, bins=50, density=True)
        plt.xlabel('R$^2$')
        plt.ylabel('Density')
        plt.title('R$^2$ distribution (min = '+str(np.min(r2))+', max = '+str(np.max(r2))+', median = '+str(np.median(r2))+').')
        plt.show()

    return mse, r2


def pixel_intensity(true, predicted):
    """
    Plots the distribution of pixel intensities for the validation set (or any set made of real/input image set) and the reconstructed/predicted data set.
    """
    counts_true, bins = np.histogram(true.flatten(), bins=50)
    counts_pred, bins = np.histogram(predicted.flatten(), bins=bins)

    plt.figure('Pixel intensity distribution')
    plt.plot(counts_true, 's-', label='True')
    plt.plot(counts_pred, 's-', label='Predicted')
    plt.legend()
    plt.xlabel('Pixel intensity')
    plt.ylabel('Counts')
    plt.show()


def shear_estimation():
    """

    """

    return 0
