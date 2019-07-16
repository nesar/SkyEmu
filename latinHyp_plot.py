''' Latin hypercube design
Installation: pip install --upgrade pyDOE
https://pythonhosted.org/pyDOE/randomized.html

'''

import numpy as np
from matplotlib import pyplot as plt
import pyDOE as pyDOE
import sys
import SetPub
SetPub.set_pub()


def rescale01(xmin, xmax, f):
    return (f - xmin) / (xmax - xmin)


num_evals = 64
num_params = 7
verbose = True
np.random.seed(7)

#########################################################################
####### Parameters -- these should follow the following syntax ########
# para = np.linspace(lower_lim, upper_lim, total_eval)


# cosmology parameters - excluding tau and dark energy
para1 = np.linspace(0.12, 0.155, num_evals)  #OmegaM
para2 = np.linspace(0.0215, 0.0235, num_evals) #Omegab
para3 = np.linspace(0.7, 0.89, num_evals) # sigma8
para4 = np.linspace(0.55, 0.85, num_evals) # h
para5 = np.linspace(0.85, 1.05, num_evals) # n_s

# redshift parameters
if (num_params==7):
    para6 = np.linspace(0.5, 1.5, num_evals) # z_m
    para7 = np.linspace(0.05, 0.5, num_evals) # FWHM

# no other known option yet - can insert other options, or read from text file 
elif(num_params>5):
    print("unknown parameter option")


if (num_params==5):
    AllPara = np.vstack([para1, para2, para3, para4, para5])
    AllLabels = [r'$\tilde{\omega}_m$', r'$\tilde{\omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{'
                 r'h}$', r'$\tilde{n}_s$']
elif (num_params==7):
    AllPara = np.vstack([para1, para2, para3, para4, para5, para6, para7])
    AllLabels = [r'$\tilde{\omega}_m$', r'$\tilde{\omega}_b$', r'$\tilde{\sigma}_8$', r'$\tilde{'
                 r'h}$', r'$\tilde{n}_s$', r'$\tilde{z}_m$', r'$\tilde{FWHM}$']

#########################################################################


# latin hypercube
lhd = pyDOE.lhs(AllPara.shape[0], samples=num_evals, criterion=None) # c cm corr m


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
            print(i,j)
            if(i!=j):
                a[i, j].scatter(lhd[:, i], lhd[:, j], s = 1, alpha = 0.7)
                a[i, j].grid(True)
                a[j, i].set_visible(False)

            else:
                # a[i,i].set_title(AllLabels[i])
                a[i, i].text(0.4, 0.4, AllLabels[i], size = 'x-large')
                hist, bin_edges = np.histogram(lhd[:,i], density=True, bins=12)
                # a[i,i].bar(hist)
                a[i,i].bar(bin_edges[:-1], hist/hist.max(), width=0.09, alpha = 0.5)
                plt.xlim(0,1)
                plt.ylim(0,1)
    # plt.tight_layout()
    plt.savefig('Plots/LatinSq.pdf', figsize=(5000, 5000), bbox_inches="tight", dpi = 900)

    plt.show()


idx = (lhd * num_evals).astype(int)

AllCombinations = np.zeros((num_evals, AllPara.shape[0]))
for i in range(AllPara.shape[0]):
    AllCombinations[:, i] = AllPara[i][idx[:, i]]

np.savetxt('lhc_'+str(num_evals)+'_'+str(num_params)+'.txt', AllCombinations)   
if verbose:
    print(AllCombinations)
