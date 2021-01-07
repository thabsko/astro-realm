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

	def dgauss( x, a1, wid1, g_cen1, a2, wid2, g_cen2, cont ):
		gauss1 = a1*np.exp(-(x-g_cen1)**2 /(2*wid1**2))
		gauss2 = a2*np.exp(-(x-g_cen2)**2 /(2*wid2**2))

		return gauss1 + gauss2 + cont