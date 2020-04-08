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
		CI_path : str 
			Path for ALMA [CI] datacubes

		CI_moment0 : str
			Filename of [CI] moment-0 map

		CI_rms : 
			Mean RMS noise 

		regions : 1D array
			Names of the DS9 region files

		dl : float
			Length of distance scale bar

		source : str
			Shorthand name of source

		save_path : str
			Path for saved output
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

class HeII_analysis:

	def __init__( self,  dec, ra, size, lam1, lam2, muse_file, 
		p, wav_em, source, output_dir ):
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

	def gauss1( self, x, amp, wid, g_cen, cont ):
		"""
		Gaussian function with continuum
		Integrated flux is a fit parameter
	
		Parameters 
		----------
		x : array
			Wavelength axis
	
		amp : float
			Integrated flux
	
		wid : float
			FWHM
	
		g_cen : float
			Gaussian centre
	
		cont : float
			Continuum value
	
		Return
		------
		Gaussian function : 1d array
	
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
		y : float
			DEC (pixel) of aperture centre for extracted MUSE spectrum
	
		x : float
			RA (pixel) of aperture centre for extracted MUSE spectrum
	
		size : float
			Radius of aperture for extracted MUSE spectrum
	
		lam1 : float
			Wavelength (Angstroms) at the lower-end of spectral range 
			of the subcube
	
		lam2 : float
			Wavelength (Angstroms) at the upper-end of spectral range 
			of the subcube
	
		muse_file : str
			Path and filename of MUSE datacube
	
		p : 1d array
			Initial guesses for fit parameters
	
		wav_em : float 
			Rest wavelength of HeII 1640
	
		source : str
			Name of source
	
		save_path : str
			Path for saved output

		Returns 
		-------
		Systemic redshift of the galaxy and its velocity : 1d array
		
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
		
		mod 	= lm.Model(self.gauss1) 
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
	
		pl.plot(vel_arr, self.gauss1(wav, res['amp'], res['wid'], 
			res['g_cen'], res['cont']), c='red')
		pl.plot(vel_arr, flux, c='k', drawstyle='steps-mid')
		pl.xlim([-1500,1500])
		pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=12)
		pl.ylabel(r'F$_\lambda$ / $10^{-20}$ erg s$^{-1}$ $\AA^{-1}$ 	cm$^{-2}$', fontsize=12)
	
		pl.savefig(self.output_dir+self.source+'_HeII.png')
	
		print( "Systemic redshift ("+source+"): %.4f +/- %.4f " %( z, z_err ) 	)
		return [z, z_err, vel_glx]