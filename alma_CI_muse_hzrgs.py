# S.N. Kolwa 
# ESO (2019)

from astropy.io import fits
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
from math import*

import mpdaf.obj as mpdo
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
import astropy.units as u
from astropy.coordinates import Distance, Angle
from astropy.cosmology import Planck15

import matplotlib.ticker as tk

from itertools import chain 

from lmfit import *
import lmfit.models as lm

import pyregion as pyr
from matplotlib.patches import Ellipse

import warnings
from astropy.utils.exceptions import AstropyWarning

import pickle
import time

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ('ignore', category=AstropyWarning         )


class CI_analysis:
	
	def __init__( self, CI_path, CI_moment0, 
		CI_rms, regions, dl, source, input_dir, 
		output_dir):
		"""
		Parameters 
		----------
		CI_path : Path for ALMA [CI] datacubes

		CI_moment0 : Filename of [CI] moment-0 map

		CI_rms : Mean RMS noise 

		regions : Names of the DS9 region files

		dl : Length of distance scale bar

		source : Shorthand name of source

		input_dir : Location of input files

		output_dir : Location of output files
		"""

		self.CI_path = CI_path
		self.CI_moment0 = CI_moment0
		self.CI_rms = CI_rms
		self.regions = regions
		self.dl = dl
		self.source	= source
		self.input_dir = input_dir
		self.output_dir = output_dir

	def CI_narrow_band( self ):
		"""
		Visualise narrow-band ALMA [CI] moment-0 map generated with CASA. 

		Returns 
		-------
		Moment-0 map : image 
		
		"""
		moment0 = fits.open(self.CI_path+self.CI_moment0)
		hdr = moment0[0].header
		wcs = WCS(hdr).sub(axes=2)

		img_arr = moment0[0].data[0,0,:,:]

		img_arr = np.rot90(img_arr, 1)
		img_arr = np.flipud(img_arr)

		# Save moment-0 map with colourbar
		fig = pl.figure(figsize=(7,5))
		ax = fig.add_axes([0.02, 0.11, 0.95, 0.85], 
			projection=wcs)

		ax.set_xlabel(r'$\alpha$ (J2000)', size=14)
		ax.set_ylabel(r'$\delta$ (J2000)', size=14)

		#define contour parameters	
		n_contours 		=	4
		n 				=   1
		
		contours 		= np.zeros(n_contours)
		contours[0] 	= self.CI_rms
		
		for i in range(1,n_contours):
			contours[i] = self.CI_rms*np.sqrt(2)*n
			n			+= 1
	
		pix = list(chain(*img_arr))
		pix_rms = np.sqrt(np.mean(np.square(pix)))
		pix_med = np.median(pix)
		vmax = 2*(pix_med + pix_rms) * 1.e3
		vmin = (pix_med - pix_rms) * 1.e3

		N = img_arr.shape[0]
		CI_data = [[ img_arr[i][j]*1.e3 for i in range(N) ] for j in range(N)]	#mJy

		# Draw contours
		ax.contour(CI_data, levels=contours*1.e3, colors='blue',
	 	label='[CI](1-0)', zorder=-5)

		self.regions = [ self.input_dir+i for i in self.regions]
	
		for x in self.regions:
			r = pyr.open(x)
			patch, text = r.get_mpl_patches_texts()
	
			for p in patch:
				ax.add_patch(p)
	
			for t in text:
				ax.add_artist(t)

		# Draw synthesised beam

		# degrees per pixel
		pix_deg = abs(hdr['cdelt1'])							
		# clean beam Parameters
		bmaj, bmin, pa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  	
		ellip = Ellipse( (10,10), (bmaj/pix_deg), (bmin/pix_deg), (180-pa),
		fc='yellow', ec='black' )
		ax.add_artist(ellip)
	
		ax.text(80, 10, '10 kpc', color='red', fontsize=10, 
			bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5)
		ax.plot([83., 83.+ self.dl], [8.5, 8.5], c='red', lw='2', zorder=10.)
		ra 	= ax.coords[0]
		dec = ax.coords[1]
	
		ra.set_major_formatter('hh:mm:ss.s')
	
		img_arr = img_arr[:,:]
		N = img_arr.shape[0]
		img_arr = [[ img_arr[i][j]*1.e3 for i in range(N) ] for j in range(N)]
		CI_map = ax.imshow(img_arr, origin='lower', cmap='gray_r', 
			vmin=vmin, vmax=vmax, zorder=-10)
	
		left, bottom, width, height = ax.get_position().bounds
		cax = fig.add_axes([ left*42., 0.11, width*0.04, height ])
		cb = pl.colorbar(CI_map, orientation = 'vertical', cax=cax)
		cb.set_label(r'mJy beam$^{-1}$',rotation=90, fontsize=12)
		cb.ax.tick_params(labelsize=12)
	
		pl.savefig(self.output_dir+self.source+'_CI_moment0.png')

	def gauss_peak(self, x, a, wid, g_cen, cont ):
		"""
		Gaussian function with continuum
		Peak flux is a fit parameter
	
		Parameters 
		----------
		x : Wavelength axis
	
		a : Peak flux
	
		wid : Full-width at Half Maximum
	
		g_cen : Gaussian centre
	
		cont : Continuum level
	
		Return
		------
		Gaussian function : 1D array
		
			"""
		self.a = a
		self.wid = wid
		self.g_cen = g_cen
		self.cont = cont

		gauss = self.a*np.exp(-(x-self.g_cen)**2 /(2*self.wid**2))
		return gauss + self.cont

	def CI_H2_lum_mass(self, z, z_err, SdV, SdV_err, nu_obs, nu_obs_err):
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
		[CI] luminosity and mass, H_2 mass

		"""
		self.z = z
		self.z_err = z_err
		self.SdV = SdV
		self.SdV_err = SdV_err
		self.nu_obs = nu_obs
		self.nu_obs_err = nu_obs_err

		Dl = (Distance(z=self.z, unit=u.Mpc, cosmology=Planck15)).value
		X_CI = 3.e-5
		A_10 = 7.93e-8
		Q_10 = 0.5

		L_CI = 3.25e7*self.SdV*1.e-3*Dl**2/(nu_obs**2*(1+z)**3)  # L' in K km/s pc^2
		L_CI_err = L_CI*np.sqrt( (self.SdV_err/self.SdV)**2 + (self.nu_obs_err/self.nu_obs)**2 )

		print( 'L_CI = %.2e + %.2e' %(L_CI, L_CI_err) )

		T1 = 23.6		# energy above ground state for [CI](1-0)
		T2 = 62.5		# energy above ground state for [CI](2-1)
		T_ex = 40		# excitation temp.
		Q_Tex = 1. + 3.*e**(-T1/T_ex) + 5.*e**(-T2/T_ex)
		
		M_CI = 5.706e-4*Q_Tex*(1./3)*e**(23.6/T_ex)*L_CI 	# solar masses

		M_H2 = (1375.8*Dl**2*(1.e-5/X_CI)*(1.e-7/A_10)*SdV*1.e-3)/((1.+z)*Q_10) # solar masses

		M_H2_err = M_H2*( (self.z_err/self.z)**2 + (self.SdV_err/self.SdV)**2 )

		print( 'M_CI <= %.2e' %M_CI )

		print( 'M_H2/M_sol = %.3e +/- %.3e' %(M_H2, M_H2_err) )
		print( 'M_H2/M_sol (upper limit) <= %.3e' %M_H2 )

		return M_H2

	def CI_host_galaxy( self, CI_datacube, z, z_err,
		SFR, mean, sig_rms ):
		"""
		Visualise narrow-band ALMA [CI] moment-0 map generated with CASA
	
		Parameters 
		----------
		CI_datacube : [CI] datacube path
			
		CI_moment0 : [CI] moment-0 map filename
			
		CI_datacube : [CI] primary-beam corrected datacube path
	
		z : Redshift of source
			
		z_err : Redshift error of source
		
		SFR : Star-formation rate of source 

		mean : Average noise

		sig_rms : rms noise 

		Returns 
		-------
		Flux : str
		Frequency : str
		S/N of detection : str
		
		"""	
		self.CI_datacube = CI_datacube
		hdu = fits.open(self.CI_path+self.CI_datacube)
		data = hdu[0].data[0,0,:,:]			#mJy/beam
	
		# mask host galaxy region
		c = 50
		for i,j in zip(range(c-2,c+2),range(c-2,c+2)):
			data[i,j] = None
	
		cells = [ list(chain(*data[i:j, i:j])) for i,j in zip(range(0,100,1),	range(1,100,1)) ]
	
		# remove cells with NaN (masked) values
		N = len(cells)
	
		remove = []
		for i in range(N):
			for j in range(1):
				if isnan(cells[i][j]) == 1:
					remove.append(i)
	
		cells 	= np.delete(cells, remove, axis=0)
	
		N 		= len(cells)
		sum_cells = [ cells[i].sum() for i in range(N) ]
		
		pl.figure(figsize=(10,5))
	
		hist = pl.hist(sum_cells, color='blue', bins=int(len(cells)/2))
		
		med = np.median(hist[1])
		std = np.std(hist[1])
	
		print("Median: %.2f mJy"%(med *1.e3)) 
		print("Std Dev: %.2f mJy"%(std *1.e3))
	
		# Fit distribution
		pars = Parameters()
		pars.add_many( 
			('g_cen', med, True, -1., 1.),
			('a', 10., True, 0.),	
			('wid', 0.0001, True, 0.),  	
			('cont', 0., True, 0., 0.5 ))
	
		mod 	= lm.Model(self.gauss_peak) 
		x = [ hist[1][i] + (hist[1][i+1] - hist[1][i])/2 for i in range(len(hist[0])) ]
	
		fit 	= mod.fit(hist[0], pars, x=x)
		pl.plot(x, fit.best_fit, 'r--' )
	
		res = fit.params
		mu =res['g_cen'].value		#Jy/beam
	
		moment0 = fits.open(self.CI_path+self.CI_moment0)
		hdr = moment0[0].header
	
		pix_deg = abs(hdr['cdelt1'])
		bmaj, bmin = hdr['bmaj'], hdr['bmin']
		bmaj, bmin = bmaj*0.4/pix_deg, bmin*0.4/pix_deg		# arcsec^2
		print('bmaj: %.2f arcsec, bmin: %.2f arcsec' %(bmaj, bmin))
	
		print(fit.fit_report())
		pl.xlabel(r'Jy beam$^{-1}$')
		pl.ylabel('N')
		pl.savefig(self.output_dir+self.source+'_field_flux_hist.png')
	
		# flux of host galaxy 
		alma_spec = np.genfromtxt(self.output_dir+self.source+'_spec_host_freq.txt')
	
		freq = alma_spec[:,0]	# GHz
		flux = alma_spec[:,1]	# Jy  (flux extracted from 1 beam-sized area)
		
		M = len(freq)
	
		freq_e = 491.161
		v_radio = [ c*(1. - freq[i]/freq_e) for i in range(M) ] 
	
		freq_o = freq_e/(1.+z)			# frequency at the systemic redshift (	from HeII)
		freq_o_err = freq_o*( z_err/z )
		vel0 = c*(1 - freq_o/freq_e)	# systemic velocity
	
		voff = [ v_radio[i] - vel0 for i in range(M) ]	# offset from systemic 	velocity
		
		# get indices for data between -25 and 25 km/s
		flux_sub = [ flux[i] for i in range(M) if (voff[i] < 80. and voff[i] > 80) ]
		flux_sub = np.array(flux_sub)
	
		# host galaxy detection above flux noise (field)
		sigma, sigma_err = res['wid'].value, res['wid'].stderr
	
		freq_host = freq_e/(1+z)
		freq_host_err = (z_err/z)*freq_host
	
		SdV = 3.*sig_rms*1.e3*100.										# flux 	in mJy km/s 
		M_H2 = self.CI_H2_lum_mass(z, z_err, SdV, 0., freq_o, freq_o_err)	# H_2 	mass in solar masses
		tau_depl = M_H2/SFR  	#depletion time-scale in yr
	
		print("tau_depl. = %.2f" %(tau_depl/1.e6))	#depletion time-scale in Myr
		print(r'SdV = %.2f mJy.km/s' %SdV)		
		print('M_H2/M_sol < %.2e' %M_H2) 	
		print(r'Frequency of host galaxy: $\nu$ =  %.2f +/- %.2f' %(freq_host, freq_host_err))
 	
		return [ mu*1.e3, sigma*1.e3 ]


class HeII_analysis:

	def __init__( self,  dec, ra, size, lam1, lam2, muse_file, 
		p, wav_em, source, output_dir ):
		"""
		Parameters 
		----------
		dec : Declination (pixel) of aperture centre
		
		ra : Right Ascension (pixel) of aperture centre

		size : Radius of aperture 

		lam1 : Lower wavelength

		lam2 : Upper wavelength

		muse_file : MUSE datacube path

		p : Initial parameters for line-fit

		wav_em : Rest wavelength of line

		source : Source name

		output_dir : Location of output files

		""" 
		self.dec = dec
		self.ra = ra
		self.size = size
		self.lam1 = lam1
		self.lam2 = lam2
		self.muse_file = muse_file
		self.p = p
		self.wav_em = wav_em
		self.source = source
		self.output_dir = output_dir

		self.c = 2.9979245800e5 		#speed of light in km/s

	def gauss_int( self, x, amp, wid, g_cen, cont ):
		"""
		Gaussian function with continuum
		Integrated flux is a fit parameter
	
		Parameters 
		----------
		x : Wavelength axis
	
		amp : Integrated flux
	
		wid : Full-width at Half-Maximum
	
		g_cen : Gaussian centre
	
		cont : Continuum level
	
		Return
		------
		Gaussian function : 1D array
	
		"""
		self.amp = amp
		self.wid = wid
		self.g_cen = g_cen
		self.cont = cont

		gauss = (self.amp/np.sqrt(2.*np.pi)/self.wid) * np.exp(-(x-self.g_cen)**2 /(2.*self.wid**2))

		return gauss + self.cont

	def wav_to_vel( self, wav_obs, wav_em, z ):
		"""
		Convert an observed wavelength to a velocity 
		in the observer-frame at redshift
	
		Parameters 
		----------
		wav_obs : Observed wavelength
	
		wav_em : Emitted wavelength
	
		z : Observer Redshift
	
		Returns 
		-------
		velocity (km/s)
		
		"""
		self.wav_obs = wav_obs
		self.wav_em = wav_em
		self.z = z
		v = self.c*((self.wav_obs/self.wav_em/(1.+self.z)) - 1.)
		return v

	def muse_redshift( self ):
		"""
		Calculate the systemic redshift from a line in the MUSE spectrum
	
		Parameters 
		----------
		y : Declination (pixel) of aperture centre 
	
		x : Right Ascension (pixel) of aperture centre 
	
		size : Radius of aperture for extracted MUSE spectrum
	
		lam1 : Wavelength (Angstroms) at the lower-end of spectral range 
			of the subcube
	
		lam2 : Wavelength (Angstroms) at the upper-end of spectral range 
			of the subcube
	
		muse_file : Path and filename of MUSE datacube
	
		p : Initial guesses for fit parameters
	
		wav_em : Rest wavelength of HeII 1640
	
		source : Name of source
	
		save_path : Path for saved output

		Returns 
		-------
		Systemic redshift of the galaxy and its velocity : 1D array
		
		"""
		muse_cube = mpdo.Cube(self.muse_file, ext=1)
		
		m1,m2 	= muse_cube.sum(axis=(1,2)).wave.pixel([self.lam1,self.lam2], nearest=True) 
		muse_cube = muse_cube[ m1:m2, :, : ]
		sub = muse_cube.subcube_circle_aperture( (self.dec, self.ra), self.size, 	unit_center=None, unit_radius=None )
		
		fig = pl.figure(figsize=(6,5))
		fig.add_axes([0.12, 0.1, 0.85, 0.85])
		ax = pl.gca()
		spec = sub.sum(axis=(1,2))
		wav = spec.wave.coord()
		flux = spec.data
	
		pars = Parameters()
		pars.add_many( 
			('g_cen', self.p[0], True, self.p[0] - 5., self.p[0] + 5.),
			('amp', self.p[1], True, 0.),	
			('wid', self.p[2], True, 0.),  	#GHz
			('cont', self.p[3], True ))
		
		mod 	= lm.Model(self.gauss_int) 
		fit 	= mod.fit(flux, pars, x=wav)
	
		res = fit.params
	
		wav_obs = res['g_cen'].value

		z = (wav_obs/self.wav_em - 1.) 

		if res['g_cen'].stderr is not None:
			z_err = ( res['g_cen'].stderr / res['g_cen'].value )*z

		else:
			z_err = 0.

		vel_glx = self.c*z					#velocity (ref frame of observer)
	
		vel_arr = [ self.wav_to_vel( wav[i], self.wav_em, z ) for i in range(len(wav)) 	] # at z=z_sys
	
		pl.plot(vel_arr, self.gauss_int(wav, res['amp'], res['wid'], 
			res['g_cen'], res['cont']), c='red')
		pl.plot(vel_arr, flux, c='k', drawstyle='steps-mid')
		pl.xlim([-1500,1500])
		pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=12)
		pl.ylabel(r'F$_\lambda$ / $10^{-20}$ erg s$^{-1}$ $\AA^{-1}$ 	cm$^{-2}$', fontsize=12)
	
		pl.savefig(self.output_dir+self.source+'_HeII.png')
	
		print( "Systemic redshift ("+self.source+"): %.4f +/- %.4f " %( z, z_err ) 	)
		return [z, z_err, vel_glx]
