"""
S.N. Kolwa
ESO (2019) 

"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as pl
from math import*

from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import Distance, Angle
from astropy.cosmology import Planck15

from itertools import chain 

from lmfit import *
import lmfit.models as lm

import pyregion as pyr
from matplotlib.patches import Ellipse

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ('ignore', category=AstropyWarning)


class Image_CI:
	
	def __init__( self, CI_path=None, input_dir=None, output_dir=None ):
		"""
		Parameters 
		----------
		CI_path : Path of ALMA [CI] datacubes

		input_dir : Directory of input files

		output_dir : Directory of output files

		"""

		self.CI_path = CI_path
		self.input_dir = input_dir
		self.output_dir = output_dir

	def make_narrow_band( self, CI_moment0, 
		CI_rms, regions, dl, source ):
		"""
		Visualise narrow-band ALMA [CI] moment-0 map generated with CASA. 

		Parameters 
		----------
		CI_moment0 : [CI] moment-0 map 

		CI_rms : Minimum threshold value of [CI] contours

		regions : Region names 

		dl : Length of distance scale bar

		source : Source name
		

		Returns 
		-------
		Moment-0 map : image
		
		"""
		# Moment-0 map from CASA
		moment0 = fits.open(self.CI_path+CI_moment0)

		# WCS header
		hdr = moment0[0].header
		wcs = WCS(hdr)
		wcs = wcs.sub(axes=2)

		# Greyscale image of data
		img_arr = moment0[0].data[0,0,:,:]

		img_arr = np.rot90(img_arr, 1)
		img_arr = np.flipud(img_arr)

		# Optimise image colour-scale
		pix = list(chain(*img_arr))
		pix_rms = np.sqrt(np.mean(np.square(pix)))
		pix_med = np.median(pix)
		vmax = 2.0*(pix_med + pix_rms) * 1.e3
		vmin = 0.02*(pix_med - pix_rms) * 1.e3

		# Convert from Jy/beam to mJy/beam
		img_arr = img_arr[:,:]
		N1,N2 = img_arr.shape[0], img_arr.shape[1]
		img_arr = [[ img_arr[i][j]*1.e3 for i in range(N1) ] for j in range(N2)]

		# Save moment-0 map with colourbar
		fig = pl.figure(figsize=(7,5))
		ax = fig.add_axes([0.02, 0.11, 0.95, 0.85], projection=wcs)

		# Add contours
		[CI_data, CI_wcs, ci_contours] = self.CI_contours(self.CI_path, CI_moment0, CI_rms)

		ax.contour(img_arr, levels=ci_contours*1.e3, colors='blue',
		 label='[CI](1-0)', zorder=-5)

		# Annotate regions
		regions = [ self.input_dir+i for i in regions]
		
		for x in regions:
			r = pyr.open(x)
			patch, text = r.get_mpl_patches_texts()

			for p in patch:
				ax.add_patch(p)

			for t in text:
				ax.add_artist(t)

		# Add clean/synthesised beam
		pix_deg = abs(hdr['cdelt1'])							# degrees per pixel
		bmaj, bmin, pa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  	# clean beam Parameters
		ellip = Ellipse( (10,10), (bmaj/pix_deg), (bmin/pix_deg), (180-pa),
		fc='yellow', ec='black' )
		ax.add_artist(ellip)

		# Add projected distance-scale
		ax.text(80, 10, '10 kpc', color='red', fontsize=10, 
			bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5)
		ax.plot([83., 83.+dl], [8.5, 8.5], c='red', lw='2', zorder=10.)

		# Change format of right-ascension units
		ra 	= ax.coords[0]
		ra.set_major_formatter('hh:mm:ss.s')

		CI_map = ax.imshow(img_arr, origin='lower', cmap='gray_r', 
			vmin=vmin, vmax=vmax, zorder=-10)

		ax.set_xlabel(r'$\alpha$ (J2000)', size=14)
		ax.set_ylabel(r'$\delta$ (J2000)', size=14)

		cb = pl.colorbar(CI_map, orientation = 'vertical')
		cb.set_label('mJy/beam',rotation=90, fontsize=14)
		cb.ax.tick_params(labelsize=12)

		return pl.savefig(self.output_dir+source+'_CI_moment0.png')

	def get_mass( self, z, z_err, SdV, SdV_err, nu_obs, nu_obs_err ):
		"""
		Calculate L_CI, M_CI, M_H2

		Parameters 
		----------
		z : Redshift
			
		z_err : Redshift Error

		SdV : Integrated flux in mJy
			
		SdV_err : Error in integrated flux

		nu_obs : Observed Frequency

		nu_obs_err : Observed Frequency Error
			
		Returns 
		-------
		[CI] luminosity and inferred H_2 mass

		"""
		Dl = (Distance(z=z, unit=u.Mpc, cosmology=Planck15)).value
		X_CI = 3.e-5
		A_10 = 7.93e-8
		Q_10 = 0.5

		L_CI = 3.25e7*SdV*1.e-3*Dl**2/(nu_obs**2*(1+z)**3)  # L' in K km/s pc^2
		L_CI_err = L_CI*np.sqrt( (SdV_err/SdV)**2 + (nu_obs_err/nu_obs)**2 )

		print( 'L_CI = %.2e + %.2e' %(L_CI, L_CI_err) )

		T1 = 23.6		# energy above ground state for [CI](1-0)
		T2 = 62.5		# energy above ground state for [CI](2-1)
		T_ex = 40		# excitation temp.
		Q_Tex = 1. + 3.*e**(-T1/T_ex) + 5.*e**(-T2/T_ex)
		
		M_CI = 5.706e-4*Q_Tex*(1./3)*e**(23.6/T_ex)*L_CI 	# solar masses

		M_H2 = (1375.8*Dl**2*(1.e-5/X_CI)*(1.e-7/A_10)*SdV*1.e-3)/((1.+z)*Q_10) # solar masses

		M_H2_err = M_H2*( (z_err/z)**2 + (SdV_err/SdV)**2 )

		print( 'M_CI <= %.2e' %M_CI )

		print( 'M_H2/M_sol = %.3e +/- %.3e' %(M_H2, M_H2_err) )
		print( 'M_H2/M_sol (upper limit) <= %.3e' %M_H2 )

		return M_H2

	def get_host_galaxy_params( self, CI_moment0, CI_datacube, 
		source, CI_rms, s, z, z_err, input_dir ):
		"""
		Visualise narrow-band ALMA [CI] moment-0 map generated with CASA
	
		Parameters 
		----------
			
		CI_moment0 : [CI] moment-0 map 
			
		CI_datacube : [CI] primary-beam corrected datacube
	
		source : Source name

		CI_rms : Minimum threshold value of [CI] contours

		s : SFR and SFR error

		z : Redshift of source
			
		z_err : Redshift error of source

		input_dir : Directory of input files


		Returns 
		-------
		Flux : str
		Frequency : str
		S/N of detection : str
		
		"""	
		c = 2.9979245800e5 	#speed of light in km/s
		freq_em = 492.161		#rest frequency of [CI](1-0) in GHz

		print("-"*len("   "+source+" Host Galaxy   "))
		print("   "+source+" Host Galaxy   ")
		print("-"*len("   "+source+" Host Galaxy   "))


		hdu = fits.open(self.CI_path+CI_datacube)
		data = hdu[0].data[0,:,:,:]
		hdr = hdu[0].header	
	
		pix = data.flatten()
	
		N = len(pix)
		mean = np.mean(pix)*1.e3	# average intensity
		std = np.std(pix)			# standard deviation intensity
		sqrs = [ pix[i]**2 for i in range(N) ]	
		rms = np.sqrt( sum(sqrs) / N  )*1.e3
	
		m0_hdu = fits.open(self.CI_path+CI_moment0)
		m0_hdr = m0_hdu[0].header
	
		# Synthesized beam-size
		pix_deg = abs(m0_hdr['cdelt1'])
		bmaj, bmin = m0_hdr['bmaj'], m0_hdr['bmin']
		bmaj, bmin = bmaj*0.4/pix_deg, bmin*0.4/pix_deg			# arcsec^2
	
		print('bmaj: %.2f arcsec, bmin: %.2f arcsec' %(bmaj, bmin))
	
		# Spectrum for host galaxy
		alma_spec = np.genfromtxt(input_dir+source+'_spec_host_freq.txt')
	
		freq = alma_spec[:,0]	# GHz
		flux = alma_spec[:,1]	# Jy 
		
		M = len(freq)
	
		v_radio = [ c*(1. - freq[i]/freq_em) for i in range(M) ] 
	
		# systemic frequency
		freq_o = freq_em/(1. + z) 
		freq_o_err = freq_o*( z_err/z )
	
		# systemic velocity
		vel0 = c*(1. - freq_o/freq_em)	
	
		# velocity offsets
		voff = [ v_radio[i] - vel0 for i in range(M) ]	
		
		# host galaxy
		freq_host = freq_em/(1+z)
		freq_host_err = (z_err/z)*freq_host
	
		SdV = 3.*rms*100.		# 3*sigma flux in mJy km/s (FWHM=100 km/s)
	
		print('SdV = %.2f mJy.km/s' %SdV)	
		print('Frequency of host galaxy (from systemic z) = %.2f +/- %.2f GHz' 
			%(freq_host, freq_host_err))
	
		M_H2 = self.get_mass(z, z_err, SdV, 0., freq_o, freq_o_err)	# H_2 mass in solar masses
	
		[SFR, err_SFR_upp, err_SFR_low] = s

		try:
			print('SFR = %.0f + %.0f - %.0f' % (SFR, err_SFR_upp, err_SFR_low))
			print('tau_(depl.) = %.0f Myr' % ((M_H2/SFR)/1.e6))
	
		except: 
			print("No SFR measured")
	
		return [ mean, rms ]

	def CI_contours( self, CI_path, CI_moment0, CI_rms ):
		"""
		Reads header and image data and
		generates [CI] contours from moment-0 map
	
		Parameters 
		----------
		CI_path : Path of ALMA [CI] datacubes
	
		CI_moment0 : [CI] moment-0 map 
	
		CI_rms : Minimum threshold value of [CI] contours
	
		Return
		------
		[CI] image array, WCS and contours : 1d array
	
		"""
		moment0 = fits.open(CI_path+CI_moment0)
		CI_wcs	= WCS(moment0[0].header)
	
		# select RA,DEC axes only 
		CI_new_wcs = CI_wcs.sub(axes = 2)	
		CI_img_arr = moment0[0].data[0,0,:,:]
		
		#define contour parameters	
		n_contours 		=	4
		n 				=   1
		
		contours 		= np.zeros(n_contours)
		contours[0] 	= CI_rms
		
		for i in range(1,n_contours):
			contours[i] = CI_rms*np.sqrt(2)*n
			n			+= 1
	
		return [ CI_img_arr, CI_new_wcs, contours ]

