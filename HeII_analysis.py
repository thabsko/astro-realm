# S.N. Kolwa 
# ESO (2019)

import numpy as np
import matplotlib.pyplot as pl
from math import*

import mpdaf.obj as mpdo

from lmfit import *
import lmfit.models as lm

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ('ignore', category=AstropyWarning         )


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
	
		# if source=='TNJ1338':
		# 	pl.ylim(-100, 40.)
	
		pl.savefig(self.output_dir+self.source+'_HeII.png')
	
		print( "Systemic redshift ("+self.source+"): %.4f +/- %.4f " %( z, z_err ) 	)
		return [z, z_err, vel_glx]
