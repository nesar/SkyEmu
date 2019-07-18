''' Latin hypercube design
Installation: pip install --upgrade pyDOE
https://pythonhosted.org/pyDOE/randomized.html

'''

import numpy as np
from matplotlib import pyplot as plt
import pyDOE2 as pyDOE
import sys
# import SetPub
# SetPub.set_pub()


def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


num_evals = 512
num_params = 5
verbose = True
np.random.seed(7)

#########################################################################
####### Parameters -- these should follow the following syntax ########
# para = np.linspace(lower_lim, upper_lim, total_eval)


# cosmology parameters - excluding tau and dark energy
para1 = np.linspace(1e4, 1e5, num_evals)  # Flux
para2 = np.linspace(0.1, 1., num_evals)  # Radius
para3 = np.linspace(-0.5, 0.5, num_evals)  # g1
para4 = np.linspace(-0.5, 0.5, num_evals)  # g2
para5 = np.linspace(0.2, 0.4, num_evals)  # psf fwhm

# redshift parameters
if (num_params == 7):
    para6 = np.linspace(0.5, 1.5, num_evals)  # z_m
    para7 = np.linspace(0.05, 0.5, num_evals)  # FWHM

# no other known option yet - can insert other options, or read from text file
elif(num_params > 5):
    print("unknown parameter option")


if (num_params == 5):
    AllPara = np.vstack([para1, para2, para3, para4, para5])
    AllLabels = [r'Flux', r'Radius', r'Shear g1', r'Shear g2', r'PSF fwhm']
elif (num_params == 7):
    AllPara = np.vstack([para1, para2, para3, para4, para5, para6, para7])
    AllLabels = [r'$\tilde{\omega}_m$', r'$\tilde{\omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{'
                 r'h}$', r'$\tilde{n}_s$', r'$\tilde{z}_m$', r'$\tilde{FWHM}$']

#########################################################################


# latin hypercube
lhd = pyDOE.lhs(AllPara.shape[0], samples=num_evals, criterion=None)  # c cm corr m


if verbose:
    print(lhd)
# lhd = norm(loc=0, scale=1).ppf(lhd)  # this applies to both factors here

##
if verbose:
    f, a = plt.subplots(AllPara.shape[0], AllPara.shape[0], sharex=True, sharey=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.rcParams.update({'font.size': 4})

    for i in range(AllPara.shape[0]):
        for j in range(i+1):
            print(i, j)
            if(i != j):
                a[i, j].scatter(lhd[:, i], lhd[:, j], s=1, alpha=0.7)
                a[i, j].grid(True)
                a[j, i].set_visible(False)

            else:
                # a[i,i].set_title(AllLabels[i])
                a[i, i].text(0.4, 0.4, AllLabels[i], size='x-large')
                hist, bin_edges = np.histogram(lhd[:, i], density=True, bins=12)
                # a[i,i].bar(hist)
                a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
    # plt.tight_layout()
    # plt.savefig('/../Data/Plots/LatinSq.pdf', figsize=(5000, 5000), bbox_inches="tight", dpi=900)

    plt.show()


idx = (lhd * num_evals).astype(int)

AllCombinations = np.zeros((num_evals, AllPara.shape[0]))
for i in range(AllPara.shape[0]):
    AllCombinations[:, i] = AllPara[i][idx[:, i]]

# Delete row when g1**2 + g2**2 > 1
del_rows = np.where(AllCombinations[:, 2]**2+AllCombinations[:, 3]**2 > 1.)[0]
AllCombinations = np.delete(AllCombinations, del_rows, axis=0)

# Plot after deleting
if verbose:
    f, a = plt.subplots(AllPara.shape[0], AllPara.shape[0], sharex=True, sharey=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.rcParams.update({'font.size': 4})

    for i in range(AllCombinations.shape[1]):
        for j in range(i+1):
            print(i, j)
            if(i != j):
                a[i, j].scatter(AllCombinations[:, i], AllCombinations[:, j], s=1, alpha=0.7)
                a[i, j].grid(True)
                a[j, i].set_visible(False)

            else:
                # a[i,i].set_title(AllLabels[i])
                a[i, i].text(0.4, 0.4, AllLabels[i], size='x-large')
                hist, bin_edges = np.histogram(AllCombinations[:, i], density=True, bins=12)
                # a[i,i].bar(hist)
                a[i, i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha=0.5)
    # plt.tight_layout()
    # plt.savefig('/../Data/Plots/LatinSq_delete.pdf', figsize=(5000, 5000), bbox_inches="tight", dpi=900)
    plt.show()

np.savetxt('lhc_'+str(num_evals)+'_'+str(num_params)+'.txt', AllCombinations)

if verbose:
    print(AllCombinations)
