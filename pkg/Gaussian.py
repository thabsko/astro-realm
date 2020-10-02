"""
S.N. Kolwa
ESO (2019) 

"""

import numpy as np

class Gaussian:

	def gauss( x, a, wid, g_cen, cont ):
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