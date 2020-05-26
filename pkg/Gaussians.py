"""
S.N. Kolwa
ESO (2019) 

"""

import numpy as np

class Gaussians:

	def gauss_peak( x, a, wid, g_cen, cont ):
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
		Ordinate : list
		
		"""
		gauss = a*np.exp(-(x-g_cen)**2 /(2*wid**2))
		return gauss + cont

	def gauss_int( x, amp, wid, g_cen, cont ):
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
		Ordinate : list

		"""

		gauss = (amp/np.sqrt(2.*np.pi)/wid) * np.exp(-(x-g_cen)**2 /(2.*wid**2))
		return gauss + cont
