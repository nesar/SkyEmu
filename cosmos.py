import galsim
import numpy as np
import matplotlib.pyplot as plt
import os


def main(ind):
	nx = 33
	ny = 33
	pixel_scale = 0.44

	nimages = 10

	sky_level = 0

	if not os.path.isdir('../Data/output_cosmos'):
	    os.mkdir('../Data/output_cosmos')

	    file_name = os.path.join('../Data/output_cosmos', 'power_spectrum.fits')

	    gal_dilation = 1
	    gal_snr = 100
	    psf_snr = 1000

	# Load COSMOS catalog (use_real=False for using parametric galaxies, not the real ones).
	# Note that if you want to make real galaxies you need to download and store the full
	# tarball with all galaxy images, whereas if you want to make parametric galaxies
	# you only need the catalog real_galaxy_catalog_25.2_fits.fits (and the selection file
	# real_galaxy_catalog_25.2_selection.fits if you want to place cuts on the postage stamp quality)
	# and can delete the galaxy and PSF image files.
	# cat_file_name = 'real_galaxy_catalog_25.2_fits.fits'
	# dir = '../../GalSim-2.1.5/bin/'
	# cat = galsim.COSMOSCatalog(cat_file_name, dir=dir)
	cat = galsim.COSMOSCatalog(use_real=False)

	params_rec = cat.getParametricRecord(index=ind)
	# print(params_rec)

	for x in params_rec:
		print(x)
		print(params_rec[x])
		print('-------------------------------')

	gal = cat.makeGalaxy(index=ind)

	gal_img = gal.drawImage().array

	plt.imshow(gal_img)
	plt.show()

	# params_real = cat.getRealParams(index=0)
	# print(params_real)


	# image = cat.makeGalaxy(index=10).drawImage()

	# plt.imshow(image.array)
	# plt.show()

	# for i in range(nimages):
	#     plt.subplot(nimages//2, 2, i+1)
	#     plt.imshow(cat.makeGalaxy(i).drawImage())
	# plt.show()
