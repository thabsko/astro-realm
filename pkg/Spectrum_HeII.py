"""
S.N. Kolwa
ESO (2019) 

"""

import numpy as np
import matplotlib.pyplot as pl
from math import*

import mpdaf.obj as mpdo

from lmfit import *
import lmfit.models as lm

from Gaussians import * 

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ('ignore', category=AstropyWarning         )


class Spectrum_HeII:

	def __init__( self, source=None, output_dir=None ):
		"""
		Parameters 
		----------
		source : Source name

		output_dir : Location of output files

		""" 
	
		self.source = source
		self.output_dir = output_dir
		self.c = 2.9979245800e5 	#speed of light in km/s
		self.freq_e = 492.161		#rest frequency of [CI](1-0) in GHz

	def convert_wav_to_vel( self, wav_obs, wav_em, z ):
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

	def get_redshift( self, dec, ra, size, lam1, lam2, muse_file, 
		p, wav_em ):
		"""
		Calculate the systemic redshift from a line in the MUSE spectrum

		Parameters 
		----------
		dec : DEC (pixel) of aperture centre for extracted MUSE spectrum

		ra : RA (pixel) of aperture centre for extracted MUSE spectrum

		size : Radius of aperture for extracted MUSE spectrum

		lam1 : Wavelength (Angstroms) at the lower-end of spectral range 
			of the subcube

		lam2 : Wavelength (Angstroms) at the upper-end of spectral range 
			of the subcube

		muse_file : Path and filename of MUSE datacube

		p : Initial guesses for fit parameters

		wav_em : Rest wavelength of HeII 1640

		source : Name of source

		Returns 
		-------
		Systemic redshift of the galaxy and its velocity : list
		
		"""
		self.dec = dec
		self.ra = ra
		self.size = size
		self.lam1 = lam1
		self.lam2 = lam2
		self.muse_file = muse_file
		self.p = p
		self.wav_em = wav_em

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
		
		mod 	= lm.Model(Gaussians.gauss_int) 
		fit 	= mod.fit(flux, pars, x=wav)
	
		res = fit.params
	
		wav_obs = res['g_cen'].value

		z = (wav_obs/self.wav_em - 1.) 

		if res['g_cen'].stderr is not None:
			z_err = ( res['g_cen'].stderr / res['g_cen'].value )*z

		else:
			z_err = 0.

		vel_glx = self.c*z					#velocity (ref frame of observer)
	
		vel_arr = [ self.convert_wav_to_vel( wav[i], self.wav_em, z ) for i in range(len(wav)) 	] # at z=z_sys
	
		pl.plot(vel_arr, Gaussians.gauss_int(wav, res['amp'], res['wid'], 
			res['g_cen'], res['cont']), c='red')
		pl.plot(vel_arr, flux, c='k', drawstyle='steps-mid')
		pl.xlim([-1500,1500])
		pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=12)
		pl.ylabel(r'F$_\lambda$ / $10^{-20}$ erg s$^{-1}$ $\AA^{-1}$ 	cm$^{-2}$', fontsize=12)
	
		pl.savefig(self.output_dir+self.source+'_HeII.png')
	
		print( "Systemic redshift ("+self.source+"): %.4f +/- %.4f " %( z, z_err ) 	)
		return [z, z_err, vel_glx]

