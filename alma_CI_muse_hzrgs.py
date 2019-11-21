# S.N. Kolwa 
# ESO (2019)
# hzrg_functions.py

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

# fundamental constants
c 				= 2.9979245800e5 	#speed of light in km/s

def freq_to_vel(freq_obs, freq_em, z):
	"""
	Convert a observed frequency to a velocity 
	in the observer-frame at redshift, z

	Parameters 
	----------
	freq_obs : float	
		Observed frequency

	freq_em : float
		Rest frequency

	z : float
		Redshift of observer-frame

	Returns 
	-------
	velocity : float
	
	"""
	v = c*((freq_em/freq_obs/(1.+z)) - 1.)
	return v

def wav_to_vel(wav_obs, wav_em, z):
	"""
	Convert a observed wavelength to a velocity 
	in the observer-frame at redshift, z

	Parameters 
	----------
	freq_obs : float	
		Observed frequency

	freq_em : float
		Rest frequency

	z : float
		Redshift of observe-frame

	Returns 
	-------
	velocity : float
	
	"""
	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	return v

def convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err ):	
	"""
	Convert sigma/dispersion to FWHM (km/s)

	Parameters 
	----------
	sigma : float 
		Velocity dispersion 

	sigma_err : float
		Velocity dispersion error

	freq_o : float
		Observed frequency

	freq_o_err : float
		Observed frequency error

	Returns 
	-------
	FWHM and FWHM error (km/s) : 1d array
	
	"""

	fwhm 		= 2.*np.sqrt(2*np.log(2))*sigma  #
	fwhm_err 	= (sigma_err/sigma)*fwhm
	
	fwhm_kms	= (fwhm/492.161)*c
	fwhm_kms_err = fwhm_kms*(fwhm_err/fwhm) 
	
	return [fwhm_kms,fwhm_kms_err]

def CI_narrow_band( CI_path, CI_moment0, CI_rms, regions, dl, source, save_path ):
	"""
	Visualise narrow-band ALMA [CI] moment-0 map generated with CASA. 

	Parameters 
	----------
	CI_path : str 
		Path for ALMA [CI] datacubes

	CI_moment0 : str
		Filename of [CI] moment-0 map

	regions : 1d-array
		Names of the DS9 region files

	scale_pos : str
		RA, DEC on map where arcsec - kpc 
		distance scale conversion will be shown

	dl : float
		Length of distance scale bar

	source : str
		Shorthand name of source

	save_path : str
		Path for saved output

	Returns 
	-------
	Moment-0 map : image 
	
	"""

	# Moment-0 map from CASA
	moment0 = fits.open(CI_path+CI_moment0)

	# WCS header
	hdr = moment0[0].header
	wcs = WCS(hdr)
	wcs = wcs.sub(axes=2)
	img_arr = moment0[0].data[0,0,:,:]

	img_arr = np.rot90(img_arr, 1)
	img_arr = np.flipud(img_arr)

	# Save moment-0 map with colourbar
	fig = pl.figure(figsize=(7,5))
	ax = fig.add_axes([0.02, 0.11, 0.95, 0.85], projection=wcs)
	ax.set_xlabel(r'$\alpha$ (J2000)', size=14)
	ax.set_ylabel(r'$\delta$ (J2000)', size=14)

	CI_fn = CI_contours(CI_path, CI_moment0, CI_rms)

	CI_data, CI_wcs, ci_contours 	= CI_fn[0], CI_fn[1], CI_fn[2]

	CI_data = np.rot90(CI_data, 1)
	CI_data = np.flipud(CI_data)

	pix = list(chain(*CI_data))
	pix_rms = np.sqrt(np.mean(np.square(pix)))
	pix_med = np.median(pix)
	vmax = 2*(pix_med + pix_rms) * 1.e3
	vmin = (pix_med - pix_rms) * 1.e3

	N = CI_data.shape[0]
	CI_data = [[ CI_data[i][j]*1.e3 for i in xrange(N) ] for j in xrange(N)]	#mJy

	ci_contours = ci_contours*1.e3

	ax.contour(CI_data, levels=ci_contours, colors='blue',
	 label='[CI](1-0)', zorder=-5)

	regions = [ save_path+i for i in regions]
	
	for x in regions:
		r = pyr.open(x)
		patch, text = r.get_mpl_patches_texts()

		for p in patch:
			ax.add_patch(p)

		for t in text:
			ax.add_artist(t)

	# draw clean/synthesised beam
	pix_deg = abs(hdr['cdelt1'])							# degrees per pixel
	bmaj, bmin, pa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  	# clean beam Parameters
	ellip = Ellipse( (10,10), (bmaj/pix_deg), (bmin/pix_deg), (180-pa),
	fc='yellow', ec='black' )
	ax.add_artist(ellip)

	# # add arcsec -> kpc scale
	# c1 = SkyCoord(scale_pos, unit=(u.hourangle, u.deg))
	# pix1 = skycoord_to_pixel(c1, wcs)

	ax.text(80, 10, '10 kpc', color='red', fontsize=10, 
		bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5)
	ax.plot([83., 83.+dl], [8.5, 8.5], c='red', lw='2', zorder=10.)
	ra 	= ax.coords[0]
	dec = ax.coords[1]

	ra.set_major_formatter('hh:mm:ss.s')

	img_arr = img_arr[:,:]
	N = img_arr.shape[0]
	img_arr = [[ img_arr[i][j]*1.e3 for i in xrange(N) ] for j in xrange(N)]
	CI_map = ax.imshow(img_arr, origin='lower', cmap='gray_r', 
		vmin=vmin, vmax=vmax, zorder=-10)

	left, bottom, width, height = ax.get_position().bounds
	cax = fig.add_axes([ left*42., 0.11, width*0.04, height ])
	cb = pl.colorbar(CI_map, orientation = 'vertical', cax=cax)
	cb.set_label(r'mJy beam$^{-1}$',rotation=90, fontsize=12)
	cb.ax.tick_params(labelsize=12)

	return pl.savefig(save_path+source+'_CI_moment0.png')


def CI_host_galaxy( CI_path, CI_datacube, CI_moment0, source, z, z_err, save_path ):
	"""
	Visualise narrow-band ALMA [CI] moment-0 map generated with CASA

	Parameters 
	----------
	CI_path : str 
		Path for ALMA [CI] datacubes

	CI_moment0 : str
		Filename of [CI] moment-0 map

	CI_datacube : str
		Filename of [CI] (pbcor) datacube

	z : float
		Redshift of source (from HeII 1640 line)

	z_err : float
		Redshift error of source

	save_path : str
		Path for saved output

	Returns 
	-------
	Flux : str
	Frequency : str
	S/N of detection : str
	
	"""	
	hdu = fits.open(CI_path+CI_datacube)
	data = hdu[0].data[0,0,:,:]			#mJy/beam

	# mask host galaxy region
	c = 50
	for i,j in zip(range(c-2,c+2),range(c-2,c+2)):
		data[i,j] = None

	cells = [ list(chain(*data[i:j, i:j])) for i,j in zip(range(0,100,1),range(1,100,1)) ]

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

	hist = pl.hist(sum_cells, color='blue',bins=len(cells)/2)
	
	med = np.median(hist[1])
	std = np.std(hist[1])

	print "Median: %.2f mJy"%(med *1.e3)
	print "Std Dev: %.2f mJy"%(std *1.e3)

	# Fit distribution
	pars = Parameters()
	pars.add_many( 
		('g_cen', med, True, -1., 1.),
		('a', 10., True, 0.),	
		('wid', 0.0001, True, 0.),  	
		('cont', 0., True, 0., 0.5 ))

	mod 	= lm.Model(gauss2) 
	x = [ hist[1][i] + (hist[1][i+1] - hist[1][i])/2 for i in range(len(hist[0])) ]

	fit 	= mod.fit(hist[0], pars, x=x)
	pl.plot(x, fit.best_fit, 'r--' )

	res = fit.params
	mu =res['g_cen'].value		#Jy/beam

	moment0 = fits.open(CI_path+CI_moment0)
	hdr = moment0[0].header

	pix_deg = abs(hdr['cdelt1'])
	bmaj, bmin = hdr['bmaj'], hdr['bmin']
	bmaj, bmin = bmaj*0.4/pix_deg, bmin*0.4/pix_deg		# arcsec^2
	print 'bmaj: %.2f arcsec, bmin: %.2f arcsec' %(bmaj, bmin)

	print fit.fit_report()
	pl.xlabel(r'Jy beam$^{-1}$')
	pl.ylabel('N')
	pl.savefig(save_path+source+'_field_flux_hist.png')

	# flux of host galaxy 
	alma_spec = np.genfromtxt(save_path+source+'_spec_host_freq.txt')

	freq = alma_spec[:,0]	# GHz
	flux = alma_spec[:,1]	# Jy  (flux extracted from 1 beam-sized area)
	
	M = len(freq)

	freq_e = 491.161
	v_radio = [ c*(1. - freq[i]/freq_e) for i in range(M) ] 

	freq_o = freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	freq_o_err = freq_o*( z_err/z )
	vel0 = c*(1 - freq_o/freq_e)	# systemic velocity

	voff = [ v_radio[i] - vel0 for i in range(M) ]	# offset from systemic velocity
	
	# # get indices for data between -25 and 25 km/s
	# flux_sub = [ flux[i] for i in range(M) if (voff[i] < 80. and voff[i] > -80) ]
	# flux_sub = np.array(flux_sub)

	# host_flux = abs(flux_sub.mean())

	# host galaxy detection above flux noise (field)
	sigma, sigma_err = res['wid'].value, res['wid'].stderr

	freq_host = freq_e/(1+z)
	freq_host_err = (z_err/z)*freq_host

	if source == '4C03':
		SFR = 142.
		mean = 2.74739273e-05
		sig_rms = 0.00016965				

	elif source == '4C04':
		print "No SFR available"
		SFR = 0.
		mean = 6.87968588e-05
		sig_rms = 0.00019437

	elif source == '4C19':
		SFR = 84.
		mean = -4.16734594e-05
		sig_rms = 0.00021209

	elif source == 'MRC0943':
		SFR = 41.
		mean = -1.60191097e-06
		sig_rms = 0.00019499

	elif source == 'TN_J0121':
		SFR = 626.
		mean = -1.14772672e-05
		sig_rms = 0.00014129

	elif source == 'TNJ0205':
		SFR = 84.
		mean = -1.47495775e-05
		sig_rms = 0.00020811

	elif source == 'TNJ1338':
		SFR = 461.
		mean = 1.86797525e-05
		sig_rms = 0.00014025

	SdV = 3.*sig_rms*1.e3*100.										# flux in mJy km/s 
	M_H2 = CI_H2_lum_mass(z, z_err, SdV, 0., freq_o, freq_o_err)	# H_2 mass in solar masses
	tau_depl = M_H2/SFR  	#depletion time-scale in yr

	print "tau_depl. = %.2f" %(tau_depl/1.e6)	#depletion time-scale in Myr
	print r'SdV = %.2f mJy.km/s' %SdV 			
	print 'M_H2/M_sol < %.2e' %M_H2 		
	print r'Frequency of host galaxy (from systemic z) $\nu$ =  %.2f +/- %.2f' %(freq_host, freq_host_err)
 
	return [ mu*1.e3, sigma*1.e3 ]

def CI_spectrum_4C03( CI_path, CI_moment0, path_muse, muse_cube, 
	freq_e, z, z_err, mu, sig_rms, save_path ):
	"""
	Show 4C03 [CI] line spectra and line-fits

	Parameters 
	----------
	CI_path : str
		Path for [CI] spectrum file

	CI_moment0 : str
		Filename of [CI] moment-0 map

	path_muse : str
		Path for MUSE datacube

	muse_cube : str
		Filename of MUSE datacube

	freq_e : float
		Rest frequency of [CI] i.e. 492.161 GHz

	z : float
		Systemic redshift for source (from HeII 1640 line)

	Returns 
	-------
	[CI](1-0) spectra of 4C03.24 : image
	
	"""
	CI_spec = [ 'NW', 'host', 'E' ]		# region names	

	moment0 = fits.open(CI_path+CI_moment0)
	hdr =  moment0[0].header

	bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees

	freq_e = freq_e/1.e9

	for CI_spec in CI_spec:

		# frequency (GHz) array of the spectrum
		alma_spec = np.genfromtxt('../4C03_output/4C03_spec_'+CI_spec+'_freq.txt')

		freq = alma_spec[:,0]	# GHz
		flux = alma_spec[:,1]	# Jy			
		flux = [ alma_spec[:,1][i]*1.e3 for i in range(len(flux)) ] 	# mJy

		alma_spec_80 = np.genfromtxt('../4C03_output/4C03_spec_host_80kms_freq.txt')

		freq_80 = alma_spec_80[:,0]	
		flux_80 = alma_spec_80[:,1]
		flux_80 = [ flux_80[i]*1.e3 for i in range(len(flux_80)) ] 	

		# draw spectrum
		fig = pl.figure(figsize=(7,5))
		fig.add_axes([0.15, 0.1, 0.8, 0.8])
		ax = pl.gca()

		if CI_spec in ('E','NW'):
			print '------------'
			print '  '+CI_spec+' 4C03'
			print '------------'

			# Fit [CI] line
			pars = Parameters()
			pars.add_many( 
				('g_cen', 107.8, True, 0.),
				('a', 0.35, True, 0.),	
				('wid', 0.03, True, 0.),  	
				('cont', 0.05, True ))
			
			mod 	= lm.Model(gauss2) 
			fit 	= mod.fit(flux, pars, x=freq)

			print fit.fit_report()
			
			res = fit.params

			sigma, sigma_err 	= res['wid'].value, res['wid'].stderr
			freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr

			sigma_kms = (sigma/freq_e)*c

			sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )

			fwhm_kms, fwhm_kms_err = convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )

			print "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err)

			print "FWHM (km/s) = %.3f +/- %.3f" %(fwhm_kms, fwhm_kms_err)

			amp, amp_err 	= res['a'].value, res['a'].stderr	
			flux_peak 		= amp		 		# peak flux in mJy
			flux_peak_err	= amp_err     		# peak flux in mJy

			SdV 		= flux_peak * fwhm_kms 				# integrated flux in mJy km/s
			SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )

			significance =  (flux_peak *(fwhm_kms/20.)) / sig_rms

			print "Flux peak (mJy) = %.3f +/- %.3f" %(flux_peak, flux_peak_err)

			print "S/N = %.2f "%significance

			print "Flux integrated (mJy.km/s) = %.3f +/- %.3f" %(SdV, SdV_err)

			M_H2 = CI_H2_lum_mass(z, z_err, SdV, SdV_err, freq_o, freq_o_err)	

			freq_sys = freq_e/(1.+z)

			v_sys = c*(1. - freq_sys/freq_e)

			v_obs = c*(1. - freq_o/freq_e)

			vel_offset = v_obs - v_sys
			vel_offset_err = vel_offset*( (freq_o_err/freq_o)**2 + (z_err/z)**2 )

			print "Velocity offset (km/s) = %.3f +/- %.3f" %( vel_offset, vel_offset_err )

			SFR = 142.

			tau_depl = M_H2/SFR  	#yr

			print "tau_depl. = %.2f" %(tau_depl/1.e6)	#Myr
	 		
			freq_ax = np.linspace( freq.min(), freq.max(), num=100)
			ax.plot(freq_ax, gauss2(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

			fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

			np.savetxt(save_path+CI_spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

		else:
			flux = flux

		if CI_spec == 'E':
			E_cont = np.genfromtxt(save_path+'E_fit_params.txt')[6]

		else:
			N_cont = np.genfromtxt(save_path+'NW_fit_params.txt')[6]

		fs = 10
		ax.set_xlabel(r'$\nu_{\rm obs}$ (GHz)', fontsize=fs)
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.tick_params('both', labelsize=fs)
		ax.plot( freq, flux, c='k', drawstyle='steps-mid' )
		ax.plot( freq_80, flux_80, c='#0a78d1', drawstyle='steps-mid' )

		pl.savefig(save_path+CI_spec+'_CI_spectrum.png')

		pickle.dump( fig, file( save_path+'CI_spec_'+CI_spec+'.pickle', 'wb' ) )

	mpl_ax1 = pickle.load(open(save_path+'CI_spec_host.pickle', 'rb'))
	data1 = mpl_ax1.axes[0].lines[0].get_data()			#data chan=20 km/s
	data1_80 = mpl_ax1.axes[0].lines[1].get_data() 		#data chan=80 km/s
	
	mpl_ax2 = pickle.load(open(save_path+'CI_spec_NW.pickle', 'rb'))
	data2_1 = mpl_ax2.axes[0].lines[0].get_data()		#model
	data2_2 = mpl_ax2.axes[0].lines[1].get_data()		#data
	
	mpl_ax3 = pickle.load(open(save_path+'CI_spec_E.pickle', 'rb'))
	data3_1 = mpl_ax3.axes[0].lines[0].get_data()		#model
	data3_2 = mpl_ax3.axes[0].lines[1].get_data()		#data
	
	fig, ax = pl.subplots(3, 1, figsize=(6, 9), sharex=True, constrained_layout=True)
	pl.subplots_adjust(hspace=0, wspace=0.01) 
	
	# freq to velocity shift on abcissa
	v_radio1 = [ c*(1. - data2_1[0][i]/freq_e) for i in range(len(data2_1[0])) ] 	#model 
	v_radio2 = [ c*(1. - data2_2[0][i]/freq_e) for i in range(len(data2_2[0])) ] 	#data chan=20 km/s
	v_radio3 = [ c*(1. - data1_80[0][i]/freq_e) for i in range(len(data1_80[0])) ] #data chan=80 km/s
 	
	freq_o 	= freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	vel0 	= c*(1 - freq_o/freq_e)	# systemic velocity
	
	# offset from systemic velocity
	voff1 = [ v_radio1[i] - vel0 for i in range(len(data2_1[0])) ]	
	voff2 = [ v_radio2[i] - vel0 for i in range(len(data2_2[0])) ]	
	voff3 = [ v_radio3[i] - vel0 for i in range(len(data1_80[0])) ]
	
	fs = 14
	dx = 0.05
	dy = 0.92

	ax[0].plot( voff2, data1[1], c='k', drawstyle='steps-mid' )
	ax[0].plot( voff3, data1_80[1], c='#0a78d1', drawstyle='steps-mid', lw=2 )
	ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs)
	ax[0].set_ylim([ 1.19*min(data1[1]), 1.19*max(data1[1]) ])	
	
	ax[1].plot(voff1, data2_1[1], c='red')
	ax[1].plot(voff2, data2_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 30.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data2_2_sub = [ data2_2[1][i] for i in indices ]
	ax[1].fill_between( voff2_sub, data2_2_sub, N_cont, interpolate=1, color='yellow' )
	ax[1].text(dx, dy, '(b) North', ha='left', transform=ax[1].transAxes, fontsize=fs)
	ax[1].set_ylim([ 1.19*min(data2_2[1]), 1.19*max(data2_2[1]) ])	

	ax[2].plot( voff1, data3_1[1], c='red' )
	ax[2].plot( voff2, data3_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 50.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data3_2_sub = [ data3_2[1][i] for i in indices ]
	ax[2].fill_between( voff2_sub, data3_2_sub, E_cont, interpolate=1, color='yellow' )
	ax[2].text(dx, dy, '(c) East', ha='left', transform=ax[2].transAxes, fontsize=fs)
	ax[2].set_ylim([ 1.19*min(data3_2[1]), 1.19*max(data3_2[1]) ])	
	
	for ax in ax:
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.plot([0.,0.], [-0.69, 0.69], c='gray', ls='--')

	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
	pl.savefig(save_path+'4C03_CI_spectrums.png')

def CI_spectrum_4C04( CI_path, CI_moment0, path_muse, muse_cube, 
	freq_e, z, z_err, mu, sig_rms, save_path ):
	"""
	Show 4C04 [CI] line spectra and line-fits

	Parameters 
	----------
	CI_path : str
		Path for [CI] spectrum file

	CI_moment0 : str
		Filename of [CI] moment-0 map

	path_muse : str
		Path for MUSE datacube

	muse_cube : str
		Filename of MUSE datacube

	freq_e : float
		Rest frequency of [CI] i.e. 492.161 GHz

	z : float
		Systemic redshift for source (from HeII 1640 line)

	Returns 
	-------
	[CI](1-0) spectra of 4C04.11 : image
	
	"""
	CI_spec = [ 'NE', 'host'  ]		# region names	

	moment0 = fits.open(CI_path+CI_moment0)
	hdr =  moment0[0].header

	bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees

	freq_e = freq_e/1.e9

	for CI_spec in CI_spec:
		
		# frequency (GHz) array of same spectrum
		alma_spec = np.genfromtxt('../4C04_output/4C04_spec_'+CI_spec+'_freq.txt')

		freq 		= alma_spec[:,0]	# GHz
		flux 		= alma_spec[:,1]	# Jy
		flux = [ flux[i]*1.e3 for i in range(len(flux)) ]  #mJy
		
		alma_spec = np.genfromtxt('../4C04_output/4C04_spec_host_80kms_freq.txt')

		freq_80 		= alma_spec[:,0]	# GHz
		flux_80 		= alma_spec[:,1]	# Jy
		flux_80 = [ flux_80[i]*1.e3 for i in range(len(flux_80)) ]  #mJy

		# # draw spectrum
		fig = pl.figure(figsize=(5,5))
		fig.add_axes([0.15, 0.1, 0.8, 0.8])
		ax = pl.gca()

		if CI_spec != 'host':
			print '------------'
			print '  '+CI_spec+' 4C04'
			print '------------'

			# Fit [CI] line
			pars = Parameters()
			pars.add_many( 
				('g_cen', 89.2, True, 0.),
				('a', 0.35, True, 0.),	
				('wid', 0.03, True, 0.),  	
				('cont', 0.05, True ))
			
			mod 	= lm.Model(gauss2) 
			fit 	= mod.fit(flux, pars, x=freq)

			print fit.fit_report()
			
			res = fit.params

			sigma, sigma_err 	= res['wid'].value, res['wid'].stderr
			freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr

			sigma_kms = (sigma / freq_e)*c 		

			sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )
			fwhm_kms, fwhm_kms_err = convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )

			print "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err)

			print "FWHM (km/s) = %.3f +/- %.3f" %(fwhm_kms, fwhm_kms_err)

			amp, amp_err 	= res['a'].value, res['a'].stderr	
			flux_peak 		= amp		 		# in mJy
			flux_peak_err	= amp_err     		# in mJy

			SdV 		= flux_peak * fwhm_kms 				# integrated flux in mJy km/s
			SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )

			significance =  (flux_peak *(fwhm_kms/20.)) / sig_rms

			print "Flux peak (mJy)= %.3f +/- %.3f" %(flux_peak, flux_peak_err)

			print "S/N = %.2f "%significance

			print "Flux integrated (mJy.km/s)= %.3f +/- %.3f" %(SdV, SdV_err)

			M_H2 = CI_H2_lum_mass(z, z_err, SdV, SdV_err, freq_o, freq_o_err)	

			freq_sys = freq_e/(1.+z)

			v_sys = c*(1. - freq_sys/freq_e)

			v_obs = c*(1. - freq_o/freq_e)

			vel_offset = v_obs - v_sys
			vel_offset_err = vel_offset*( (freq_o_err/freq_o)**2 + (z_err/z)**2 )

			print "Velocity offset: %.3f +/- %.3f" %( vel_offset, vel_offset_err )
	 		
			freq_ax = np.linspace( freq.min(), freq.max(), num=100)
			ax.plot(freq_ax, gauss2(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

			fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

			np.savetxt(save_path+CI_spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

		else:
			flux = flux

		if CI_spec == 'NE':
			NE_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		fs = 10
		ax.set_xlabel(r'$\nu_{\rm obs}$ (GHz)', fontsize=fs)
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.tick_params('both', labelsize=fs)
		ax.plot( freq, flux, c='k', drawstyle='steps-mid' )
		ax.plot(freq_80, flux_80, c='#0a78d1', drawstyle='steps-mid', lw=2)

		pl.savefig(save_path+CI_spec+'_CI_spectrum.png')

		pickle.dump( fig, file( save_path+'CI_spec_'+CI_spec+'.pickle', 'wb' ) )

	mpl_ax1 = pickle.load(open(save_path+'CI_spec_host.pickle', 'rb'))
	data1 = mpl_ax1.axes[0].lines[0].get_data()
	data1_80 = mpl_ax1.axes[0].lines[1].get_data()

	mpl_ax2 = pickle.load(open(save_path+'CI_spec_NE.pickle', 'rb'))
	data2_1 = mpl_ax2.axes[0].lines[0].get_data()		#model
	data2_2 = mpl_ax2.axes[0].lines[1].get_data()		#data
	
	fig, ax = pl.subplots(2, 1, figsize=(6, 6), sharex=True, constrained_layout=True)
	pl.subplots_adjust(hspace=0, wspace=0.01) 
	
	# freq to velocity shift on abcissa
	v_radio1 = [ c*(1. - data2_1[0][i]/freq_e) for i in range(len(data2_1[0])) ] 	#km/s
	v_radio2 = [ c*(1. - data2_2[0][i]/freq_e) for i in range(len(data2_2[0])) ] 	#km/s
	v_radio3 = [ c*(1. - data1_80[0][i]/freq_e) for i in range(len(data1_80[0])) ]
	
	freq_o 	= freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	vel0 	= c*(1 - freq_o/freq_e)	# systemic velocity
	
	voff1 = [ v_radio1[i] - vel0 for i in range(len(data2_1[0])) ]	# offset from systemic velocity
	voff2 = [ v_radio2[i] - vel0 for i in range(len(data2_2[0])) ]	
	voff3 = [ v_radio3[i] - vel0 for i in range(len(data1_80[0])) ]
	
	# fs = 14
	dx = 0.05
	dy = 0.92

	ax[0].plot( voff2, data1[1], c='k', drawstyle='steps-mid' )
	ax[0].plot( voff3, data1_80[1], c='#0a78d1', drawstyle='steps-mid', lw=2)
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 50.)  ]
	ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs)
	ax[0].set_ylim([ 1.19*min(data1[1]), 1.19*max(data1[1]) ])	
	
	ax[1].plot(voff1, data2_1[1], c='red')
	ax[1].plot(voff2, data2_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -10. and voff2[i] < 70.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data2_2_sub = [ data2_2[1][i] for i in indices ]
	ax[1].fill_between( voff2_sub, data2_2_sub, NE_cont, interpolate=1, color='yellow' )
	ax[1].text(dx, dy, '(b) North-east', ha='left', transform=ax[1].transAxes, fontsize=fs)
	ax[1].set_ylim([ 1.19*min(data2_2[1]), 1.19*max(data2_2[1]) ])	
	
	for ax in ax:
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')

	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
	pl.savefig(save_path+'4C04_CI_spectrums.png')

def CI_spectrum_MRC0943( CI_path, CI_moment0, path_muse, muse_cube, 
	freq_e, z, z_err, mu, sig_rms, save_path ):
	"""
	Show MRC0943-242 [CI] line spectra and line-fits

	Parameters 
	----------
	CI_path : str
		Path for [CI] spectrum file

	CI_moment0 : str
		Filename of [CI] moment-0 map

	path_muse : str
		Path for MUSE datacube

	muse_cube : str
		Filename of MUSE datacube

	freq_e : float
		Rest frequency of [CI] i.e. 492.161 GHz

	z : float
		Systemic redshift for source (from HeII 1640 line)

	Returns 
	-------
	[CI](1-0) spectra of MRC0943-242 : image
	
	"""
	CI_spec = [ 'SW2', 'NW', 'host' ]		# region names

	moment0 = fits.open(CI_path+CI_moment0)
	hdr =  moment0[0].header	

	bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees

	freq_e = freq_e/1.e9

	for CI_spec in CI_spec:

		# frequency (GHz) array of same spectrum
		alma_spec = np.genfromtxt('../0943_output/MRC0943_spec_'+CI_spec+'_freq.txt')

		freq 		= alma_spec[:,0]	# GHz
		flux 		= alma_spec[:,1]	# Jy

		flux = [ flux[i]*1.e3 for i in range(len(flux)) ]  #mJy

		alma_spec = np.genfromtxt('../0943_output/MRC0943_spec_host_80kms_freq.txt')

		freq_80 		= alma_spec[:,0]	# GHz
		flux_80 		= alma_spec[:,1]	# Jy

		flux_80 = [ flux_80[i]*1.e3 for i in range(len(flux_80)) ]  #mJy

		# # draw spectrum
		fig = pl.figure(figsize=(5,5))
		fig.add_axes([0.15, 0.1, 0.8, 0.8])
		ax = pl.gca()

		if CI_spec != 'host':
			print '------------'
			print '  '+CI_spec+' MRC0943'
			print '------------'

			# Fit [CI] line
			pars = Parameters()
			pars.add_many( 
				('g_cen', 125.25, True, 0.),
				('a', 0.4, True, 0.),	
				('wid', 0.04, True, 0.),  	
				('cont', 0.1, True ))
			
			mod 	= lm.Model(gauss2) 
			weights = [sig_rms]*len(flux)
			fit 	= mod.fit(flux, pars, x=freq, weights=weights)		# add rms 

			print fit.fit_report()
			
			res = fit.params

			sigma, sigma_err 	= res['wid'].value, res['wid'].stderr
			freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr

			sigma_kms = (sigma/(freq_e))*c 		

			sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )
			fwhm_kms, fwhm_kms_err = convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )

			print "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err)

			print "FWHM (km/s) = %.3f +/- %.3f" %(fwhm_kms, fwhm_kms_err)

			amp, amp_err 	= res['a'].value, res['a'].stderr	
			flux_peak 		= amp		 		# in mJy
			flux_peak_err	= amp_err     		# in mJy

			SdV 		= flux_peak * fwhm_kms 				# integrated flux in mJy km/s
			SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )

			signal_to_noise =  SdV / sig_rms

			print "Flux peak (mJy)= %.3f +/- %.3f" %(flux_peak, flux_peak_err)

			print "S/N = %.2f "%signal_to_noise

			print "Flux integrated (mJy.km/s)= %.3f +/- %.3f" %(SdV, SdV_err)

			M_H2 = CI_H2_lum_mass(z, z_err, SdV, SdV_err, freq_o, freq_o_err)	

			freq_sys = freq_e/(1.+z)

			v_sys = c*(1. - freq_sys/freq_e)

			v_obs = c*(1. - freq_o/freq_e)

			vel_offset = v_obs - v_sys
			vel_offset_err = vel_offset*np.sqrt( (freq_o_err/freq_o)**2 + (z_err/z)**2 )

			print "Velocity offset: %.3f +/- %.3f" %( vel_offset, vel_offset_err )

			SFR = 41.				# sol_mass / yr

			tau_depl = M_H2/SFR  	# yr

			print "tau_depl. = %.2f" %(tau_depl/1.e6)	#Myr
	 		
			freq_ax = np.linspace( freq.min(), freq.max(), num=100)
			ax.plot(freq_ax, gauss2(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

			fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

			np.savetxt(save_path+CI_spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

		else:
			flux = flux

		if CI_spec == 'NW':
			NW_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'SW2': 
			SW_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'SE':
			SE_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		fs = 10
		ax.set_xlabel(r'$\nu_{\rm obs}$ (GHz)', fontsize=fs)
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.tick_params('both', labelsize=fs)
		ax.plot( freq, flux, c='k', drawstyle='steps-mid', lw=2 )
		ax.plot( freq_80, flux_80, c='#0a78d1', drawstyle='steps-mid' )

		pl.savefig(save_path+CI_spec+'_CI_spectrum.png')

		pickle.dump( fig, file( save_path+'CI_spec_'+CI_spec+'.pickle', 'wb' ) )

	mpl_ax1 = pickle.load(open(save_path+'CI_spec_host.pickle', 'rb'))
	data1 = mpl_ax1.axes[0].lines[0].get_data()
	data1_80 = mpl_ax1.axes[0].lines[1].get_data()

	mpl_ax2 = pickle.load(open(save_path+'CI_spec_NW.pickle', 'rb'))
	data2_1 = mpl_ax2.axes[0].lines[0].get_data()		#model
	data2_2 = mpl_ax2.axes[0].lines[1].get_data()		#data

	mpl_ax3 = pickle.load(open(save_path+'CI_spec_SW2.pickle', 'rb'))
	data3_1 = mpl_ax3.axes[0].lines[0].get_data()		#model
	data3_2 = mpl_ax3.axes[0].lines[1].get_data()		#data

	fig, ax = pl.subplots(3, 1,figsize=(6, 9), sharex=True, constrained_layout=True)
	pl.subplots_adjust(hspace=0, wspace=0.01) 
	
	# freq to velocity shift on abcissa
	v_radio1 = [ c*(1. - data2_1[0][i]/freq_e) for i in range(len(data2_1[0])) ] 	#km/s
	v_radio2 = [ c*(1. - data2_2[0][i]/freq_e) for i in range(len(data2_2[0])) ] 
	v_radio3 = [ c*(1. - data1_80[0][i]/freq_e) for i in range(len(data1_80[0])) ] 	
	
	freq_o = freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	vel0 = c*(1 - freq_o/freq_e)	# systemic velocity
	
	# offset from systemic velocity
	voff1 = [ v_radio1[i] - vel0 for i in range(len(data2_1[0])) ]	
	voff2 = [ v_radio2[i] - vel0 for i in range(len(data2_2[0])) ]	
	voff3 = [ v_radio3[i] - vel0 for i in range(len(data1_80[0])) ]
	
	fs = 14
	dx = 0.05
	dy = 0.92

	ax[0].plot( voff2, data1[1], c='k', drawstyle='steps-mid' )
	ax[0].plot( voff3, data1_80[1], c='#0a78d1', drawstyle='steps-mid', lw=2)
	ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs)
	ax[0].set_ylim([ 1.19*min(data1[1]), 1.19*max(data1[1]) ])	
	
	ax[1].plot(voff1, data2_1[1], c='red')
	ax[1].plot(voff2, data2_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > 20. and voff2[i] < 100.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data2_2_sub = [ data2_2[1][i] for i in indices ]
	ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
	ax[1].text(dx, dy, '(b) North-west', ha='left', transform=ax[1].transAxes, fontsize=fs)
	ax[1].set_ylim([ 1.19*min(data2_2[1]), 1.19*max(data2_2[1]) ])	

	ax[2].plot(voff1, data3_1[1], c='red')
	ax[2].plot(voff2, data3_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > 20. and voff2[i] < 120.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data3_2_sub = [ data3_2[1][i] for i in indices ]
	ax[2].fill_between( voff2_sub, data3_2_sub, SW_cont, interpolate=1, color='yellow' )
	ax[2].text(dx, dy, '(c) South-west', ha='left', transform=ax[2].transAxes, fontsize=fs)
	ax[2].set_ylim([ 1.19*min(data3_2[1]), 1.19*max(data3_2[1]) ])	

	# ax[3].plot(voff1, data4_1[1], c='red')
	# ax[3].plot(voff2, data4_2[1], c='k', drawstyle='steps-mid')
	# indices = [ i for i in range(len(voff2)) if (voff2[i] > 20. and voff2[i] < 160.)  ]
	# voff2_sub = [ voff2[i] for i in indices ]
	# data4_2_sub = [ data4_2[1][i] for i in indices ]
	# ax[3].fill_between( voff2_sub, data4_2_sub, SE_cont, interpolate=1, color='yellow' )
	# ax[3].text(dx, dy, '(d) South-east', ha='left', transform=ax[3].transAxes, fontsize=fs)
	# ax[3].set_ylim([ 1.19*min(data4_2[1]), 1.19*max(data4_2[1]) ])	

	for ax in ax:
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')

	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
	pl.savefig(save_path+'MRC0943_CI_spectrums.png')

def CI_spectrum_J0121( CI_path, CI_moment0, path_muse, muse_cube, 
	freq_e, z, z_err, mu, sig_rms, save_path ):
	"""
	Show TN J0121+1320 [CI] line spectra and line-fits

	Parameters 
	----------
	CI_path : str
		Path for [CI] spectrum file

	CI_moment0 : str
		Filename of [CI] moment-0 map

	path_muse : str
		Path for MUSE datacube

	muse_cube : str
		Filename of MUSE datacube

	freq_e : float
		Rest frequency of [CI] i.e. 492.161 GHz

	z : float
		Systemic redshift for source (from HeII 1640 line)

	Returns 
	-------
	[CI] spectra of TN J0121+1320 : image
	
	"""
	CI_spec = [ 'NW_near', 'host', 'NW_far' ]		# region names	

	moment0 = fits.open(CI_path+CI_moment0)
	hdr =  moment0[0].header	

	bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees

	freq_e = freq_e/1.e9

	for CI_spec in CI_spec:
		# frequency (GHz) array of same spectrum
		alma_spec = np.genfromtxt('../TN_J0121_output/TN_J0121_spec_'+CI_spec+'_freq.txt')

		freq 		= alma_spec[:,0]	# GHz
		flux 		= alma_spec[:,1]	# Jy/beam

		N = len(flux)
		flux = [ flux[i]*1.e3 for i in range(N) ]  #mJy/beam
		
		# # draw spectrum
		fig = pl.figure(figsize=(5,5))
		fig.add_axes([0.15, 0.1, 0.8, 0.8])
		ax = pl.gca()

		if CI_spec in ('host', 'NW_near', 'NW_far'):
			print '------------'
			print '  '+CI_spec+' TN_J0121'
			print '------------'

			# Fit [CI] line
			pars = Parameters()
			pars.add_many( 
				('g_cen', 108.93, True, 0.),
				('a', 0.4, True, 0.),	
				('wid', 0.04, True, 0.),  	
				('cont', 0.1, True ))
			
			mod 	= lm.Model(gauss2) 
			fit 	= mod.fit(flux, pars, x=freq)

			print fit.fit_report()
			
			res = fit.params

			sigma, sigma_err 	= res['wid'].value, res['wid'].stderr
			freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr

			sigma_kms = (sigma / (freq_e*1.e-9))*c 		

			sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )
			fwhm_kms, fwhm_kms_err = convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )

			print "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err)

			print "FWHM (km/s) = %.3f +/- %.3f" %(fwhm_kms, fwhm_kms_err)

			amp, amp_err 	= res['a'].value, res['a'].stderr	
			flux_peak 		= amp		 		# in mJy
			flux_peak_err	= amp_err     		# in mJy

			SdV 		= flux_peak * fwhm_kms 				# integrated flux in mJy km/s
			SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )

			print "Flux peak (mJy) = %.3f +/- %.3f" %(flux_peak, flux_peak_err)

			print "Flux integrated (mJy.km/s)= %.3f +/- %.3f" %(SdV, SdV_err)

			M_H2 = CI_H2_lum_mass(z, z_err, SdV, SdV_err, freq_o, freq_o_err)	

			freq_sys = freq_e/(1.+z)

			v_sys = c*(1. - freq_sys/freq_e)

			v_obs = c*(1. - freq_o/freq_e)

			vel_offset = v_obs - v_sys
			vel_offset_err = vel_offset*( (freq_o_err/freq_o)**2 + (z_err/z)**2 )

			print "Velocity offset: %.3f +/- %.3f" %( vel_offset, vel_offset_err )

			SFR = 626.				# sol_mass / yr

			tau_depl = M_H2/SFR  	#yr

			print "tau_depl. = %.2f" %(tau_depl/1.e6)	#Myr
	 		
			freq_ax = np.linspace( freq.min(), freq.max(), num=100)
			ax.plot(freq_ax, gauss2(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

			fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

			np.savetxt(save_path+CI_spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

		else:
			flux = flux

		if CI_spec == 'NW_far':
			NW_far_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'host':
			host_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		fs = 10
		ax.set_xlabel(r'$\nu_{\rm obs}$ (GHz)', fontsize=fs)
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.tick_params('both', labelsize=fs)
		ax.plot( freq, flux, c='k', drawstyle='steps-mid' )

		pl.savefig(save_path+CI_spec+'_CI_spectrum.png')

		pickle.dump( fig, file( save_path+'CI_spec_'+CI_spec+'.pickle', 'wb' ) )

	mpl_ax1 = pickle.load(open(save_path+'CI_spec_host.pickle', 'rb'))
	data1_1 = mpl_ax1.axes[0].lines[0].get_data()		#model
	data1_2 = mpl_ax1.axes[0].lines[1].get_data()		#data

	mpl_ax2 = pickle.load(open(save_path+'CI_spec_NW_far.pickle', 'rb'))
	data2_1 = mpl_ax2.axes[0].lines[0].get_data()		#model
	data2_2 = mpl_ax2.axes[0].lines[1].get_data()		#data

	fig, ax = pl.subplots(2, 1,figsize=(6, 6), sharex=True, constrained_layout=True)
	pl.subplots_adjust(hspace=0, wspace=0.01) 
	
	# freq to velocity shift on abcissa
	v_radio1 = [ c*(1. - data1_1[0][i]/freq_e) for i in range(len(data1_1[0])) ] 	#km/s
	v_radio2 = [ c*(1. - data1_2[0][i]/freq_e) for i in range(len(data1_2[0])) ] 	#km/s
	
	freq_o = freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	vel0 = c*(1 - freq_o/freq_e)	# systemic velocity
	
	voff1 = [ v_radio1[i] - vel0 for i in range(len(data1_1[0])) ]	# offset from systemic velocity
	voff2 = [ v_radio2[i] - vel0 for i in range(len(data1_2[0])) ]	# offset from systemic velocity
	
	fs = 14
	dx = 0.05
	dy = 0.92
	
	# ax[0].plot(voff1, data1_1[1], c='red')
	# ax[0].plot(voff2, data1_2[1], c='k', drawstyle='steps-mid')
	# indices = [ i for i in range(len(voff2)) if (voff2[i] > -140. and voff2[i] < 200.)  ]
	# voff2_sub = [ voff2[i] for i in indices ]
	# data1_2_sub = [ data1_2[1][i] for i in indices ]
	# ax[0].fill_between( voff2_sub, data1_2_sub, NW_near_cont, interpolate=1, color='yellow' )
	# ax[0].text(dx, dy, '(a) North-west ISM', ha='left', transform=ax[0].transAxes, fontsize=fs)
	# ax[0].set_ylim([ 1.19*min(data1_2[1]), 1.19*max(data1_2[1]) ])	


	ax[0].plot(voff1, data1_1[1], c='red')
	ax[0].plot(voff2, data1_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -60. and voff2[i] < 200.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data1_2_sub = [ data1_2[1][i] for i in indices ]
	ax[0].fill_between( voff2_sub, data1_2_sub, host_cont, interpolate=1, color='yellow' )
	ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs)
	ax[0].set_ylim([ 1.19*min(data1_2[1]), 1.19*max(data1_2[1]) ])

	ax[1].plot(voff1, data2_1[1], c='red')
	ax[1].plot(voff2, data2_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -60. and voff2[i] < 80.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data2_2_sub = [ data2_2[1][i] for i in indices ]
	ax[1].fill_between( voff2_sub, data2_2_sub, NW_far_cont, interpolate=1, color='yellow' )
	ax[1].text(dx, dy, '(b) North-west CGM', ha='left', transform=ax[1].transAxes, fontsize=fs)
	ax[1].set_ylim([ 1.19*min(data2_2[1]), 1.19*max(data2_2[1]) ])	

	for ax in ax:
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')

	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
	pl.savefig(save_path+'J0121_CI_spectrums.png')

def CI_spectrum_4C19( CI_path, CI_moment0, path_muse, muse_cube, 
	freq_e, z, z_err, mu, sig_rms, save_path ):
	"""
	Show 4C19.71 [CI] line spectra and line-fits

	Parameters 
	----------
	CI_path : str
		Path for [CI] spectrum file

	CI_moment0 : str
		Filename of [CI] moment-0 map

	path_muse : str
		Path for MUSE datacube

	muse_cube : str
		Filename of MUSE datacube

	freq_e : float
		Rest frequency of [CI] i.e. 492.161 GHz

	z : float
		Systemic redshift for source (from HeII 1640 line)

	Returns 
	-------
	[CI] spectra of 4C19.71 : str
	
	"""
	CI_spec = [ 'NW', 'host', 'SE' ]		# region names	

	moment0 = fits.open(CI_path+CI_moment0)
	hdr =  moment0[0].header

	bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees
	print 'bmaj: %.2e, bmin: %.2e' %(bmaj*1.e-3, bmin*1.e-3)	

	freq_e = freq_e / 1.e9

	for CI_spec in CI_spec:

		# frequency (GHz) array of same spectrum
		alma_spec = np.genfromtxt('../4C19_output/4C19_spec_'+CI_spec+'_freq.txt')

		freq 		= alma_spec[:,0]	# GHz
		flux 		= alma_spec[:,1]	# Jy
		flux = [ flux[i]*1.e3 for i in range(len(flux)) ]  #mJy

		alma_spec_80 = np.genfromtxt('../4C19_output/4C19_spec_host_80kms_freq.txt')

		freq_80 = alma_spec_80[:,0]	
		flux_80 = alma_spec_80[:,1]
		flux_80 = [ flux_80[i]*1.e3 for i in range(len(flux_80)) ] 	
		
		# # draw spectrum
		fig = pl.figure(figsize=(5,5))
		fig.add_axes([0.15, 0.1, 0.8, 0.8])
		ax = pl.gca()

		if CI_spec in ('NW', 'SE'):
			print '------------'
			print '  '+CI_spec+' 4C19'
			print '------------'

			# Fit [CI] line
			pars = Parameters()
			pars.add_many( 
				('g_cen', 107.21, True, 0.),
				('a', 0.5, True, 0.),	
				('wid', 0.04, True, 0.),  	
				('cont', 0.05, True ))
			
			mod 	= lm.Model(gauss2) 
			fit 	= mod.fit(flux, pars, x=freq)

			print fit.fit_report()
			
			res = fit.params

			sigma, sigma_err 	= res['wid'].value, res['wid'].stderr
			freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr

			sigma_kms = (sigma / (freq_e*1.e-9))*c 		

			sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )
			fwhm_kms, fwhm_kms_err = convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )

			print "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err)

			print "FWHM (km/s) = %.3f +/- %.3f" %(fwhm_kms, fwhm_kms_err)

			amp, amp_err 	= res['a'].value, res['a'].stderr	
			flux_peak 		= amp		 		# in mJy
			flux_peak_err	= amp_err     		# in mJy

			SdV 		= flux_peak * fwhm_kms 				# integrated flux in mJy km/s
			SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )

			significance =  (flux_peak *(fwhm_kms/20.)) / sig_rms

			print "Flux peak (mJy)= %.3f +/- %.3f" %(flux_peak, flux_peak_err)

			print "S/N = %.2f "%significance

			print "Flux integrated (mJy.km/s)= %.3f +/- %.3f" %(SdV, SdV_err)

			M_H2 = M_H2 = CI_H2_lum_mass(z, z_err, SdV, SdV_err, freq_o, freq_o_err)	

			freq_sys = freq_e/(1.+z)

			v_sys = c*(1. - freq_sys/freq_e)

			v_obs = c*(1. - freq_o/freq_e)

			vel_offset = v_obs - v_sys
			vel_offset_err = vel_offset*( (freq_o_err/freq_o)**2 + (z_err/z)**2 )

			print "Velocity offset: %.3f +/- %.3f" %( vel_offset, vel_offset_err )

			SFR = 84.				# sol_mass / yr

			tau_depl = M_H2/SFR  	#yr

			print "tau_depl. = %.2f" %(tau_depl/1.e6)	#Myr
	 		
			freq_ax = np.linspace( freq.min(), freq.max(), num=100)
			ax.plot(freq_ax, gauss2(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

			fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

			np.savetxt(save_path+CI_spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')	

		else:
			flux = flux

		if CI_spec == 'SE':
			SE_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'NW':
			NW_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		fs = 10
		ax.set_xlabel(r'$\nu_{\rm obs}$ (GHz)', fontsize=fs)
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.tick_params('both', labelsize=fs)
		ax.plot( freq, flux, c='k', drawstyle='steps-mid' )
		ax.plot( freq_80, flux_80, c='#0a78d1', drawstyle='steps-mid', lw=2 )

		pl.savefig(save_path+CI_spec+'_CI_spectrum.png')

		pickle.dump( fig, file( save_path+'CI_spec_'+CI_spec+'.pickle', 'wb' ) )

	mpl_ax1 = pickle.load(open(save_path+'CI_spec_host.pickle', 'rb'))
	data1 	= mpl_ax1.axes[0].lines[0].get_data()		#data
	data1_80 = mpl_ax1.axes[0].lines[1].get_data()		#data

	mpl_ax2 = pickle.load(open(save_path+'CI_spec_NW.pickle', 'rb'))
	data2_1 = mpl_ax2.axes[0].lines[0].get_data()		#model
	data2_2 = mpl_ax2.axes[0].lines[1].get_data()		#data

	mpl_ax3 = pickle.load(open(save_path+'CI_spec_SE.pickle', 'rb'))
	data3_1 = mpl_ax3.axes[0].lines[0].get_data()		#model
	data3_2 = mpl_ax3.axes[0].lines[1].get_data()		#data

	fig, ax = pl.subplots(3, 1, figsize=(6, 9), sharex=True, constrained_layout=True)
	pl.subplots_adjust(hspace=0, wspace=0.01) 
	
	# freq to velocity shift on abcissa
	v_radio1 = [ c*(1. - data2_1[0][i]/freq_e) for i in range(len(data2_1[0])) ] 	#km/s
	v_radio2 = [ c*(1. - data2_2[0][i]/freq_e) for i in range(len(data2_2[0])) ] 	
	v_radio3 = [ c*(1. - data1_80[0][i]/freq_e) for i in range(len(data1_80[0])) ]
	
	freq_o = freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	vel0 = c*(1 - freq_o/freq_e)	# systemic velocity
	
	# offset from systemic velocity
	voff1 = [ v_radio1[i] - vel0 for i in range(len(data2_1[0])) ]	
	voff2 = [ v_radio2[i] - vel0 for i in range(len(data2_2[0])) ]	
	voff3 = [ v_radio3[i] - vel0 for i in range(len(data1_80[0])) ]	
	
	fs = 14
	dx = 0.05
	dy = 0.92

	ax[0].plot( voff2, data1[1], c='k', drawstyle='steps-mid' )
	ax[0].plot( voff3, data1_80[1], c='#0a78d1', drawstyle='steps-mid' )
	ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs)
	ax[0].set_ylim([ 1.19*min(data1[1]), 1.19*max(data1[1]) ])	
	
	ax[1].plot(voff1, data2_1[1], c='red')
	ax[1].plot(voff2, data2_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 40.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data2_2_sub = [ data2_2[1][i] for i in indices ]
	ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
	ax[1].text(dx, dy, '(b) North-west', ha='left', transform=ax[1].transAxes, fontsize=fs)
	ax[1].set_ylim([ 1.19*min(data2_2[1]), 1.19*max(data2_2[1]) ])	

	ax[2].plot(voff1, data3_1[1], c='red')
	ax[2].plot(voff2, data3_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 40.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data3_2_sub = [ data3_2[1][i] for i in indices ]
	ax[2].fill_between( voff2_sub, data3_2_sub, SE_cont, interpolate=1, color='yellow' )
	ax[2].text(dx, dy, '(c) South-east', ha='left', transform=ax[2].transAxes, fontsize=fs)
	ax[2].set_ylim([ 1.19*min(data3_2[1]), 1.19*max(data3_2[1]) ])

	for ax in ax:
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')

	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
	pl.savefig(save_path+'4C19_CI_spectrums.png')

def CI_spectrum_J0205( CI_path, CI_moment0, path_muse, muse_cube, 
	freq_e, z, z_err, mu, sig_rms, save_path ):
	"""
	Show TN J0205+2422 [CI] line spectra and line-fits

	Parameters 
	----------
	CI_path : str
		Path for [CI] spectrum file

	CI_moment0 : str
		Filename of [CI] moment-0 map

	path_muse : str
		Path for MUSE datacube

	muse_cube : str
		Filename of MUSE datacube

	freq_e : float
		Rest frequency of [CI] i.e. 492.161 GHz

	z : float
		Systemic redshift for source (from HeII 1640 line)

	Returns 
	-------
	[CI] spectra of TN J0205+2422 : str
	
	"""
	CI_spec = ['SE','host', 'E', 'W_far', 'NW']		# region names	

	moment0 = fits.open(CI_path+CI_moment0)
	hdr =  moment0[0].header

	bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees
	print 'bmaj: %.2e, bmin: %.2e' %(bmaj*1.e-3, bmin*1.e-3)

	freq_e = freq_e/1.e9

	for CI_spec in CI_spec:

		# frequency (GHz) array of same spectrum
		alma_spec = np.genfromtxt('../TN_J0205_output/TNJ0205_spec_'+CI_spec+'_freq.txt')

		freq 		= alma_spec[:,0]	# GHz
		flux 		= alma_spec[:,1]	# Jy
		flux 		= [ flux[i]*1.e3 for i in range(len(flux)) ]

		alma_spec = np.genfromtxt('../TN_J0205_output/TNJ0205_spec_host_80kms_freq.txt')

		freq_80 	= alma_spec[:,0]	# GHz
		flux_80 	= alma_spec[:,1]	# Jy
		flux_80 	= [ flux_80[i]*1.e3 for i in range(len(flux_80)) ]
		
		# # draw spectrum
		fig = pl.figure(figsize=(5,5))
		fig.add_axes([0.15, 0.1, 0.8, 0.8])
		ax = pl.gca()

		if CI_spec in ('SE', 'E', 'W_far', 'NW'):
			print '------------'
			print '  '+CI_spec+' TN_J0205'
			print '------------'

			# Fit [CI] line
			pars = Parameters()
			pars.add_many( 
				('g_cen', 109.21, True, 0.),
				('a', 0.5, True, 0., 0.8),	
				('wid', 0.04, True, 0.),  	
				('cont', 0.05, True ))
			
			mod 	= lm.Model(gauss2) 
			fit 	= mod.fit(flux, pars, x=freq)

			print fit.fit_report()
			
			res = fit.params

			sigma, sigma_err 	= res['wid'].value, res['wid'].stderr
			freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr

			sigma_kms = (sigma / (freq_e*1.e-9))*c 		

			sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )
			fwhm_kms, fwhm_kms_err = convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )

			print "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err)

			print "FWHM (km/s) = %.3f +/- %.3f" %(fwhm_kms, fwhm_kms_err)

			amp, amp_err 	= res['a'].value, res['a'].stderr	
			flux_peak 		= amp		 		# in mJy
			flux_peak_err	= amp_err     		# in mJy

			SdV 		= flux_peak * fwhm_kms 				# integrated flux in mJy km/s
			SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )

			significance =  (flux_peak *(fwhm_kms/20.)) / sig_rms

			print "Flux peak (mJy)= %.3f +/- %.3f" %(flux_peak, flux_peak_err)

			print "S/N = %.2f "%significance

			print "Flux integrated (mJy.km/s)= %.3f +/- %.3f" %(SdV, SdV_err)

			M_H2 = CI_H2_lum_mass(z, z_err, SdV, SdV_err, freq_o, freq_o_err)	

			freq_sys = freq_e/(1.+z)

			v_sys = c*(1. - freq_sys/freq_e)

			v_obs = c*(1. - freq_o/freq_e)

			vel_offset = v_obs - v_sys
			vel_offset_err = vel_offset*( (freq_o_err/freq_o)**2 + (z_err/z)**2 )

			print "Velocity offset: %.3f +/- %.3f" %( vel_offset, vel_offset_err )

			SFR = 84.				# sol_mass / yr

			tau_depl = M_H2/SFR  	#yr

			print "tau_depl. < %.2f" %(tau_depl/1.e6)	#Myr
	 		
			freq_ax = np.linspace( freq.min(), freq.max(), num=100)
			ax.plot(freq_ax, gauss2(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

			fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

			np.savetxt(save_path+CI_spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

		else:
			flux = flux

		if CI_spec == 'SE':
			SE_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'E':
			E_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		fs = 10
		ax.set_xlabel(r'$\nu_{\rm obs}$ (GHz)', fontsize=fs)
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.tick_params('both', labelsize=fs)
		ax.plot( freq, flux, c='k', drawstyle='steps-mid' )
		ax.plot( freq_80, flux_80, c='#0a78d1', drawstyle='steps-mid', lw=2)

		pl.savefig(save_path+CI_spec+'_CI_spectrum.png')

		pickle.dump( fig, file( save_path+'CI_spec_'+CI_spec+'.pickle', 'wb' ) )

	mpl_ax1 = pickle.load(open(save_path+'CI_spec_host.pickle', 'rb'))
	data1 	= mpl_ax1.axes[0].lines[0].get_data()		#data
	data1_80 = mpl_ax1.axes[0].lines[1].get_data()

	mpl_ax2 = pickle.load(open(save_path+'CI_spec_E.pickle', 'rb'))
	data2_1 = mpl_ax2.axes[0].lines[0].get_data()		#model
	data2_2 = mpl_ax2.axes[0].lines[1].get_data()		#data

	mpl_ax3 = pickle.load(open(save_path+'CI_spec_SE.pickle', 'rb'))
	data3_1 = mpl_ax3.axes[0].lines[0].get_data()		#model
	data3_2 = mpl_ax3.axes[0].lines[1].get_data()		#data

	fig, ax = pl.subplots(3, 1, figsize=(6, 9), sharex=True, constrained_layout=True)
	pl.subplots_adjust(hspace=0, wspace=0.01) 
	
	# freq to velocity shift on abcissa
	v_radio1 = [ c*(1. - data2_1[0][i]/freq_e) for i in range(len(data2_1[0])) ] 	#km/s
	v_radio2 = [ c*(1. - data2_2[0][i]/freq_e) for i in range(len(data2_2[0])) ] 
	v_radio3 = [ c*(1. - data1_80[0][i]/freq_e) for i in range(len(data1_80[0])) ] 
	
	freq_o = freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	vel0 = c*(1 - freq_o/freq_e)	# systemic velocity

	# offset from systemic velocity
	voff1 = [ v_radio1[i] - vel0 for i in range(len(data2_1[0])) ]	
	voff2 = [ v_radio2[i] - vel0 for i in range(len(data2_2[0])) ]	
	voff3 = [ v_radio3[i] - vel0 for i in range(len(data1_80[0])) ]	

	fs = 14
	dx = 0.05
	dy = 0.92

	ax[0].plot(voff2, data1[1], c='k', drawstyle='steps-mid' )
	ax[0].plot(voff3, data1_80[1], c='#0a78d1', drawstyle='steps-mid', lw=2 )
	ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs)
	ax[0].set_ylim([ 1.45*min(data1[1]), 1.45*max(data1[1]) ])

	ax[1].plot(voff1, data2_1[1], c='red')
	ax[1].plot(voff2, data2_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 60.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data2_2_sub = [ data2_2[1][i] for i in indices ]
	ax[1].fill_between( voff2_sub, data2_2_sub, E_cont, interpolate=1, color='yellow' )
	ax[1].text(dx, dy, '(b) East', ha='left', transform=ax[1].transAxes, fontsize=fs)
	ax[1].set_ylim([ 1.45*min(data2_2[1]), 1.45*max(data2_2[1]) ])	

	ax[2].plot(voff1, data3_1[1], c='red')
	ax[2].plot(voff2, data3_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 80.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data3_2_sub = [ data3_2[1][i] for i in indices ]
	ax[2].fill_between( voff2_sub, data3_2_sub, SE_cont, interpolate=1, color='yellow' )
	ax[2].text(dx, dy, '(c) South-east', ha='left', transform=ax[2].transAxes, fontsize=fs)
	ax[2].set_ylim([ 1.45*min(data3_2[1]), 1.45*max(data3_2[1]) ])	

	for ax in ax:
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')

	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
	pl.savefig(save_path+'J0205_CI_spectrums.png')

def CI_spectrum_J1338( CI_path, CI_moment0, path_muse, muse_cube, 
	freq_e, z, z_err, mu, sig_rms, save_path ):
	"""
	Show TN J1338-1934 [CI] line spectra and line-fits

	Parameters 
	----------
	CI_path : str
		Path for [CI] spectrum file

	CI_moment0 : str
		Filename of [CI] moment-0 map

	path_muse : str
		Path for MUSE datacube

	muse_cube : str
		Filename of MUSE datacube

	freq_e : float
		Rest frequency of [CI] i.e. 492.161 GHz

	z : float
		Systemic redshift for source (from HeII 1640 line)

	Returns 
	-------
	[CI] spectra of TN J1338-1934 : str
	
	"""
	CI_spec = ['NE', 'NW', 'host', 'W', 'SW']		# region names

	moment0 = fits.open(CI_path+CI_moment0)
	hdr =  moment0[0].header

	bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees

	freq_e = freq_e / 1.e9
	
	for CI_spec in CI_spec:
	
		# frequency (GHz) array of same spectrum
		alma_spec = np.genfromtxt('../TN_J1338_output/TNJ1338_spec_'+CI_spec+'_freq.txt')
		freq 		= alma_spec[:,0]	# GHz
		flux 		= alma_spec[:,1]	# Jy

		flux = [ flux[i]*1.e3 for i in range(len(flux)) ]  #mJy

		alma_spec = np.genfromtxt('../TN_J1338_output/TNJ1338_spec_host_80kms_freq.txt')
		freq_80 		= alma_spec[:,0]	# GHz
		flux_80 		= alma_spec[:,1]	# Jy

		flux_80 = [ flux_80[i]*1.e3 for i in range(len(flux_80)) ]  #mJy
		
		# # draw spectrum
		fig = pl.figure(figsize=(5,5))
		fig.add_axes([0.15, 0.1, 0.8, 0.8])
		ax = pl.gca()

		if CI_spec in ('NE', 'NW', 'W', 'SW'):
			print '------------'
			print '  '+CI_spec+' TN_J1338'
			print '------------'

			# Fit [CI] line
			pars = Parameters()
			pars.add_many( 
				('g_cen', 96.4, True, 0.),
				('a', 0.5, True, 0.),	
				('wid', 0.04, True, 0.),  	
				('cont', 0.05, True ))
			
			mod 	= lm.Model(gauss2) 
			fit 	= mod.fit(flux, pars, x=freq)

			print fit.fit_report()
			
			res = fit.params

			sigma, sigma_err 	= res['wid'].value, res['wid'].stderr		# in GHz
			freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr	# in GHz

			sigma_kms = (sigma / (freq_e*1.e-9))*c 		

			sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )
			fwhm_kms, fwhm_kms_err = convert_fwhm_kms ( sigma, sigma_err, freq_o, freq_o_err )

			print "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err)

			print "FWHM (km/s) = %.3f +/- %.3f" %(fwhm_kms, fwhm_kms_err)

			amp, amp_err 	= res['a'].value, res['a'].stderr	
			flux_peak 		= amp		 		# in mJy
			flux_peak_err	= amp_err     		# in mJy

			SdV 		= flux_peak * fwhm_kms 				# integrated flux in mJy km/s
			SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )

			significance =  (flux_peak *(fwhm_kms/20.)) / sig_rms

			print "Flux peak (mJy)= %.3f +/- %.3f" %(flux_peak, flux_peak_err)

			print "S/N = %.2f "%significance

			print "Flux integrated (mJy.km/s)= %.3f +/- %.3f" %(SdV, SdV_err)

			M_H2 = CI_H2_lum_mass(z, z_err, SdV, SdV_err, freq_o, freq_o_err)	

			freq_sys = freq_e/(1.+z)

			v_sys = c*(1. - freq_sys/freq_e)

			v_obs = c*(1. - freq_o/freq_e)

			vel_offset = v_obs - v_sys
			vel_offset_err = vel_offset*( (freq_o_err/freq_o)**2 + (z_err/z)**2 )

			print "Velocity offset: %.3f +/- %.3f" %( vel_offset, vel_offset_err )

			SFR = 461.				# sol_mass / yr

			tau_depl = M_H2/SFR  	#yr

			print "tau_depl. = %.2f" %(tau_depl/1.e6)	#Myr
	 		
			freq_ax = np.linspace( freq.min(), freq.max(), num=100)
			ax.plot(freq_ax, gauss2(freq_ax, res['a'], res['wid'], 
				res['g_cen'], res['cont']), c='red')

			fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
				res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]

			np.savetxt(save_path+CI_spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

		else:
			flux = flux

		if CI_spec == 'NE':
			NE_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'NW':
			NW_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'W':
			W_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		elif CI_spec == 'SW':
			SW_cont = np.genfromtxt(save_path+CI_spec+'_fit_params.txt')[6]

		fs = 10
		ax.set_xlabel(r'$\nu_{\rm obs}$ (GHz)', fontsize=fs)
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.tick_params('both', labelsize=fs)
		ax.plot( freq, flux, c='k', drawstyle='steps-mid' )
		ax.plot( freq_80, flux_80, c='#0a78d1', drawstyle='steps-mid' )

		pl.savefig(save_path+CI_spec+'_CI_spectrum.png')

		pickle.dump( fig, file( save_path+'CI_spec_'+CI_spec+'.pickle', 'wb' ) )

	mpl_ax1 = pickle.load(open(save_path+'CI_spec_host.pickle', 'rb'))
	data1 	= mpl_ax1.axes[0].lines[0].get_data()		#data
	data1_80= mpl_ax1.axes[0].lines[1].get_data()

	mpl_ax2 = pickle.load(open(save_path+'CI_spec_NW.pickle', 'rb'))
	data2_1 = mpl_ax2.axes[0].lines[0].get_data()		#model
	data2_2 = mpl_ax2.axes[0].lines[1].get_data()		#data

	mpl_ax3 = pickle.load(open(save_path+'CI_spec_SW.pickle', 'rb'))
	data3_1 = mpl_ax3.axes[0].lines[0].get_data()		#model
	data3_2 = mpl_ax3.axes[0].lines[1].get_data()		#data

	mpl_ax4 = pickle.load(open(save_path+'CI_spec_NE.pickle', 'rb'))
	data4_1 = mpl_ax4.axes[0].lines[0].get_data()		#model
	data4_2 = mpl_ax4.axes[0].lines[1].get_data()		#data

	fig, ax = pl.subplots(4, 1,figsize=(6, 12), sharex=True, constrained_layout=True)
	pl.subplots_adjust(hspace=0, wspace=0.01) 
	
	v_radio1 = [ c*(1. - data2_1[0][i]/freq_e) for i in range(len(data2_1[0])) ] 	#km/s
	v_radio2 = [ c*(1. - data2_2[0][i]/freq_e) for i in range(len(data2_2[0])) ] 	
	v_radio3 = [ c*(1. - data1_80[0][i]/freq_e) for i in range(len(data1_80[0])) ]
	
	freq_o = freq_e/(1.+z)			# frequency at the systemic redshift (from HeII)
	vel0 = c*(1 - freq_o/freq_e)	# systemic velocity
	
	# offset from systemic velocity
	voff1 = [ v_radio1[i] - vel0 for i in range(len(data2_1[0])) ]	
	voff2 = [ v_radio2[i] - vel0 for i in range(len(data2_2[0])) ]	
	voff3 = [ v_radio3[i] - vel0 for i in range(len(data1_80[0])) ]
	
	fs = 14
	dx = 0.05
	dy = 0.92

	ax[0].plot( voff2, data1[1], c='k', drawstyle='steps-mid' )
	ax[0].plot( voff3, data1_80[1], c='#0a78d1', drawstyle='steps-mid' )
	ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs)
	ax[0].set_ylim([ 1.19*min(data1[1]), 1.19*max(data1[1]) ])	

	ax[1].plot(voff1, data2_1[1], c='red')
	ax[1].plot(voff2, data2_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 70.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data2_2_sub = [ data2_2[1][i] for i in indices ]
	ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
	ax[1].text(dx, dy, '(b) North-west', ha='left', transform=ax[1].transAxes, fontsize=fs)
	ax[1].set_ylim([ 1.19*min(data2_2[1]), 1.19*max(data2_2[1]) ])

	ax[2].plot(voff1, data3_1[1], c='red')
	ax[2].plot(voff2, data3_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -60. and voff2[i] < 60.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data3_2_sub = [ data3_2[1][i] for i in indices ]
	ax[2].fill_between( voff2_sub, data3_2_sub, SW_cont, interpolate=1, color='yellow' )
	ax[2].text(dx, dy, '(c) South-west', ha='left', transform=ax[2].transAxes, fontsize=fs)
	ax[2].set_ylim([ 1.19*min(data3_2[1]), 1.19*max(data3_2[1]) ])		

	ax[3].plot(voff1, data4_1[1], c='red')
	ax[3].plot(voff2, data4_2[1], c='k', drawstyle='steps-mid')
	indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 80.)  ]
	voff2_sub = [ voff2[i] for i in indices ]
	data4_2_sub = [ data4_2[1][i] for i in indices ]
	ax[3].fill_between( voff2_sub, data4_2_sub, NE_cont, interpolate=1, color='yellow' )
	ax[3].text(dx, dy, '(d) North-east', ha='left', transform=ax[3].transAxes, fontsize=fs)
	ax[3].set_ylim([ 1.19*min(data4_2[1]), 1.19*max(data4_2[1]) ])	

	for ax in ax:
		ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
		ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')

	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
	pl.savefig(save_path+'J1338_CI_spectrums.png')

def CI_H2_lum_mass(z, z_err, SdV, SdV_err, nu_obs, nu_obs_err):
	"""
	Calculate L_CI, M_CI, M_H2

	Parameters 
	----------
	z : float
		Redshift

	SdV : float
		Integrated flux in mJy

	SdV_err : float
		Error in integrated flux

	Returns 
	-------
	[CI] luminosity and mass, H_2 mass

	"""

	Dl = (Distance(z=z, unit=u.Mpc, cosmology=Planck15)).value
	X_CI = 3.e-5
	A_10 = 7.93e-8
	Q_10 = 0.5

	L_CI = 3.25e7*SdV*1.e-3*Dl**2/(nu_obs**2*(1+z)**3)  # L' in K km/s pc^2
	L_CI_err = L_CI*np.sqrt( (SdV_err/SdV)**2 + (nu_obs_err/nu_obs)**2 )

	print 'L_CI = %.2e + %.2e' %(L_CI, L_CI_err)

	T1 = 23.6		# energy above ground state for [CI](1-0)
	T2 = 62.5		# energy above ground state for [CI](2-1)
	T_ex = 40		# excitation temp.
	Q_Tex = 1. + 3.*e**(-T1/T_ex) + 5.*e**(-T2/T_ex)
	
	M_CI = 5.706e-4*Q_Tex*(1./3)*e**(23.6/T_ex)*L_CI 	# solar masses

	M_H2 = (1375.8*Dl**2*(1.e-5/X_CI)*(1.e-7/A_10)*SdV*1.e-3)/((1.+z)*Q_10) # solar masses

	M_H2_err = M_H2*( (z_err/z)**2 + (SdV_err/SdV)**2 )

	print 'M_CI <= %.2e' %M_CI

	print 'M_H2/M_sol = %.3e +/- %.3e' %(M_H2, M_H2_err)
	print 'M_H2/M_sol (upper limit) <= %.3e' %M_H2

	return M_H2

def muse_redshift( y, x, size, lam1, lam2, muse_file, p, wav_em, source, save_path ):
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
	muse_cube = mpdo.Cube(muse_file, ext=1)
	
	m1,m2 	= muse_cube.sum(axis=(1,2)).wave.pixel([lam1,lam2], nearest=True) 
	muse_cube = muse_cube[ m1:m2, :, : ]
	sub = muse_cube.subcube_circle_aperture( (y, x), size, unit_center=None, unit_radius=None )

	# pl.figure()
	# img_arp = sub.sum(axis=0)
	# img_arp.plot()
	# pl.savefig(save_path+'4C03_arp_HeII_sys_redshift_est.png')
	
	fig = pl.figure(figsize=(6,5))
	fig.add_axes([0.12, 0.1, 0.85, 0.85])
	ax = pl.gca()
	spec = sub.sum(axis=(1,2))
	wav = spec.wave.coord()
	flux = spec.data

	pars = Parameters()
	pars.add_many( 
		('g_cen', p[0], True, p[0] - 5., p[0] + 5.),
		('amp', p[1], True, 0.),	
		('wid', p[2], True, 0.),  	#GHz
		('cont', p[3], True ))
	
	mod 	= lm.Model(gauss1) 
	fit 	= mod.fit(flux, pars, x=wav)

	res = fit.params

	wav_obs = res['g_cen'].value

	z = (wav_obs/wav_em - 1.)		#redshift (ref frame of observer at z=0)
	z_err = ( res['g_cen'].stderr / res['g_cen'].value )*z

	vel_glx = c*z					#velocity (ref frame of observer)

	vel_arr = [ wav_to_vel( wav[i], wav_em, z ) for i in range(len(wav)) ] # at z=z_sys

	pl.plot(vel_arr, gauss1(wav, res['amp'], res['wid'], 
		res['g_cen'], res['cont']), c='red')
	pl.plot(vel_arr, flux, c='k', drawstyle='steps-mid')
	pl.xlim([-1500,1500])
	pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=12)
	pl.ylabel(r'F$_\lambda$ / $10^{-20}$ erg s$^{-1}$ $\AA^{-1}$ cm$^{-2}$', fontsize=12)

	if source=='TNJ1338':
		pl.ylim(-100, 40.)

	pl.savefig(save_path+source+'_HeII.png')

	print "Systemic redshift ("+source+"): %.4f +/- %.4f " %( z, z_err )
	return [z, z_err, vel_glx]

def muse_lya_spectrum( muse_file, y, x, radius, lam1, lam2, z, save_path ):
	"""
	Obtain Ly-alpha spectrum 

	Parameters 
	----------
	muse_file : str
		Path and filename of MUSE datacube

	y : float
		DEC (pixel) of aperture centre for extracted MUSE spectrum

	x : float
		RA (pixel) of aperture centre for extracted MUSE spectrum

	size : float
		Radius of aperture for extracted MUSE spectrum

	lam1 : float
		Wavelength (Angstroms) at the lower-end of spectral range 
		for subcube

	lam2 : float
		Wavelength (Angstroms) at the upper-end of spectral range 
		for subcube

	z : float
		Systemic redshift of source

	save_path : str
		Path for saved output

	Returns 
	-------
	MUSE Ly-alpha spectrm : image
	
	"""
	muse_cube = mpdo.Cube(muse_file, ext=1)
	
	m1,m2 	= muse_cube.sum(axis=(1,2)).wave.pixel([lam1,lam2], nearest=True) 
	muse_cube = muse_cube[ m1:m2, :, : ]
	sub = muse_cube.subcube_circle_aperture( (y, x), radius, unit_center=None, unit_radius=None )
	
	fig = pl.figure(figsize=(6,5))
	ax = pl.gca()
	spec = sub.sum(axis=(1,2))

	wav 	=  spec.wave.coord()  		
	flux 	=  spec.data

	wav_em = 1215.57
	vel_arr = [ c*(wav[i]/wav_em/(1.+z) - 1) for i in xrange(len(wav)) ]

	ax.plot(vel_arr, flux, c='k', drawstyle='steps-mid')
	ax.set_ylabel(r'F$_\lambda$ / $10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$')
	ax.set_xlabel(r'$\Delta v$')

	pickle.dump(fig, file(save_path+'Lya_spectrum_'+`int(y)`+'_'+`int(x)`+'.pickle', 'wb') )

	pl.savefig(save_path+'Lya_spectrum_'+`int(y)`+'_'+`int(x)`+'.png')


def CI_contours( CI_path, CI_moment0, CI_rms ):
	"""
	Reads header and image data and
	generates [CI] contours from moment-0 map

	Parameters 
	----------
	CI_path : str 
		Path for ALMA [CI] datacubes

	CI_moment0 : str
		Filename of [CI] moment-0 map

	CI_rms : float
		Minimum threshold value of [CI] contours

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

def VLA_contours( vla_path, vla_img, vla_rms ):
	"""
	Reads header and image data and
	generates VLA contours from VLA image

	Parameters 
	----------
	vla_path : str 
		Path for ALMA [CI] datacubes

	vla_img : str
		Filename of VLA image

	vla_rms : float
		Minimum threshold value of VLA contours

	Return
	------
	VLA image array, WCS and contours : 1d array

	"""
	vla_hdu 	= fits.open(vla_path+vla_img)[0]
	vla_hdr 	= vla_hdu.header
	vla_data 	= vla_hdu.data[0,0,:,:]
	
	w 			= WCS(vla_hdr)

	# if vla_img in ('rc0311vla.fits'):
	# 	#transform WCS from fk4 to fk5 (B1950 to J2000)
	# 	coord		= w.wcs.crval[0:2]
	# 	c = SkyCoord(coord[0]*u.deg, coord[1]*u.deg, frame='fk4')  
	
	# 	c_fk5 = c.transform_to('fk5')   
	# 	w.wcs.crval[0], w.wcs.crval[1] = c_fk5.ra.value, c_fk5.dec.value

	# 	vla_wcs = w.celestial

	# else:
	# 	vla_wcs = w.celestial

	vla_wcs = w.celestial
	
	#VLA contour levels
	n_contours 		=	6		
		
	contours 		= [ vla_rms ]
	
	for i in range(n_contours):
		contours.append( contours[i]*(np.sqrt(2))*2 )

	return [vla_data, vla_wcs, contours]

def muse_lya_contours(muse_rms, Lya_img):
	"""
	Generate MUSE Ly-alpha contours

	Parameters 
	----------
	muse_rms : float
		Minimum threshold value of MUSE Ly-alpha contours	

	Lya_img : str
		Filename for Ly-alpha image

	Return
	------
	MUSE Ly-alpha contours : 1d array

	"""
	muse_hdu = fits.open(Lya_img)
	muse_hdr = muse_hdu[1].header
	muse_wcs = WCS(muse_hdr).celestial
	muse_data = muse_hdu[1].data[:,:]

	#MUSE Lya contour levels
	n_contours 		= 6	
	n 				= 1	
		
	contours 		= np.zeros(n_contours)
	contours[0] 	= muse_rms
	
	for i in range(1, n_contours):
		contours[i] = muse_rms*np.sqrt(2)*n
		n			+= 2

	return [muse_data, muse_wcs, contours]

def irac_contours(irac_path, irac_img, irac_rms, n_contours):
	"""
	Reads header and image data and
	generates IRAC contours from IRAC image

	Parameters 
	----------
	irac_path : str 
		Path for IRAC path

	irac_img : str
		Filename of IRAC image

	irac_rms : float
		Minimum threshold value of IRAC contours

	n_contours: float
		Number of contours

	Return
	------
	VLA image array, WCS and contours : 1d array

	"""
	irac_hdu = fits.open(irac_path+irac_img)
	irac_hdr = irac_hdu[0].header
	irac_wcs = WCS(irac_hdr).celestial

	irac_data = irac_hdu[0].data[:,:]
		
	#IRAC contour levels 
	n 				=   1		
		
	contours 		= np.zeros(n_contours)
	contours[0] 	= irac_rms
	
	for i in range(1, n_contours):
		contours[i] = contours[0]*np.sqrt(2)*n
		n			+= 1

	return [irac_data, irac_wcs, contours]


def hst_vla_CI( vla_path, vla_img, vla_rms, CI_path, CI_moment0, CI_rms,
	hst_path, hst_img, source, dl, save_path ):
	"""
	Visualise HST narrow-band data with ALMA [CI] and 
	VLA radio contours

	Parameters 
	----------
	vla_path : str
		Path for VLA image

	vla_img : str 
		VLA  image name

	vla_rms : float
		Minimum threshold value of VLA contours
	
	CI_path : str
		Path for [CI] moment-0 map

	CI_moment0 : str
		Filename of moment-0 map

	CI_rms : float
		Minimum threshold value of [CI] contours

	hst_path : str
		Path for HST data

	hst_img : str
		Filename of HST image

	source : str
		Name of source

	dl : float
		Length of distance scale bar

	save_path : str
		Path for saved output

	Return
	------
	Multiwavelength narrow-band image of HST observation overlaid
	with ALMA-detected [CI] and VLA radio contours 

	"""
	fig 		= pl.figure(figsize=(7,5))
	
	vla_fn 		= VLA_contours(vla_path, vla_img, vla_rms)

	vla_data, vla_wcs, vla_contours = vla_fn[0],  vla_fn[1], vla_fn[2]

	CI_fn = CI_contours(CI_path, CI_moment0, CI_rms)

	CI_data, CI_wcs, contours 	= CI_fn[0], CI_fn[1], CI_fn[2]

	ax = fig.add_axes([0.02, 0.11, 0.95, 0.85], projection=CI_wcs)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize('16')

	ax.set_xlabel(r'$\alpha$ (J2000)', fontsize=14)
	ax.set_ylabel(r'$\delta$ (J2000)', fontsize=14)

	ax.contour(vla_data, levels=vla_contours, colors='green',
	 transform=ax.get_transform(vla_wcs))

	ax.contour(CI_data, levels=contours, colors='blue', label='[CI](1-0)')

	N = len(CI_data)
	ax.set_xlim(0.3*N, 0.7*N)
	ax.set_ylim(0.3*N, 0.7*N)

	hdu = fits.open(hst_path+hst_img)
	hst_hdr  = hdu[1].header
	hst_data = hdu[1].data[:,:]

	hst_wcs = WCS(hst_hdr)

	x = len(hst_data)
	photflam = 2.629814606741573E-19  #erg/s/cm^2/Ang
	hst_arr = [ [hst_data[i][j]*photflam*1.e22 for j in range(x)] for i in range(x) ]
	hst_fig = ax.imshow(hst_arr, transform=ax.get_transform(hst_wcs), origin='lower', 
		interpolation='nearest', vmin=-5., vmax=25., cmap='gist_gray_r')

	left, bottom, width, height = ax.get_position().bounds
	cax = fig.add_axes([ left*42., 0.11, width*0.04, height ])
	cb = pl.colorbar(hst_fig, orientation = 'vertical', cax=cax)
	cb.set_label(r'SB / 10$^{-22}$ erg s$^{-1}$ cm$^{-2}$',rotation=90, fontsize=12)

	# c1 = SkyCoord(scale_pos, unit=(u.hourangle, u.deg))
	# pix1 = skycoord_to_pixel(c1, CI_wcs)

	ra = ax.coords[0]
	ra.set_major_formatter('hh:mm:ss.s')

	ax.text(60, 35, '10 kpc', color='red', fontsize=10, 
		bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5)
	ax.plot([61, 61+dl], [34., 34.], c='red', lw='2', zorder=10.)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize('14')

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize('14')

	return pl.savefig(save_path+source+'_hst_CI_VLA_img.png')

def irac_vla_CI( vla_path, vla_img, vla_rms, CI_path, CI_moment0, CI_rms,
	irac_path, irac_img, source, dl, save_path ):
	"""
	
	Visualise IRAC narrow-band image with ALMA [CI] and 
	VLA radio contours

	Parameters 
	----------
	vla_path : str
		Path for VLA image

	vla_img : str
		Filename of VLA image

	vla_rms : float
		Minimum threshold of VLA contours

	CI_path : str
		Path for [CI] moment-0 map

	CI_moment0 : str
		Filename of moment-0 map

	CI_rms : float
		Minimum threshold of [CI] contours
	
	irac_path : str
		Path for IRAC image
	
	irac_img : str 
		IRAC image name

	source : str
		Name of source

	dl : float
		Length of distance scale bar  

	save_path : str
		Path for saved output

	Return
	------
	Multiwavelength narrow-band image with HST data overlaid
	with ALMA-detected [CI] and VLA radio contours 

	"""
	fig   = pl.figure(figsize=(7,5))
	
	CI_fn = CI_contours(CI_path, CI_moment0, CI_rms)

	CI_data, CI_wcs, contours 	= CI_fn[0], CI_fn[1], CI_fn[2]

	ax    = fig.add_axes([0.04, 0.1, 0.85, 0.85], projection=CI_wcs)

	ax.contour(CI_data, levels=contours, colors='blue', 
	 label='[CI](1-0)')

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize('16')

	ax.set_xlabel(r'$\alpha$ (J2000)', fontsize=14)
	ax.set_ylabel(r'$\delta$ (J2000)', fontsize=14)

	vla_fn 		= VLA_contours(vla_path, vla_img, vla_rms)

	vla_data, vla_wcs, vla_contours = vla_fn[0],  vla_fn[1], vla_fn[2]

	ax.contour(vla_data, levels=vla_contours, colors='green',
		transform=ax.get_transform(vla_wcs))

	N = len(CI_data)
	ax.set_xlim(0.25*N, 0.75*N)
	ax.set_ylim(0.25*N, 0.75*N)

	hdu = fits.open(irac_path+irac_img)[0] 
	irac_hdr  = hdu.header
	irac_data = hdu.data[:,:]

	irac_wcs = WCS(irac_hdr)

	pix = list(chain(*irac_data))
	pix = [ x for x in pix if str(x) != 'nan' ]
	pix_rms = np.sqrt(np.mean(np.square(pix)))
	pix_med = np.median(pix)
	vmax = (pix_med + pix_rms) 
	vmin = (pix_med - pix_rms) 

	if source=='4C04':
		vmin, vmax = 0.1, vmax

	elif source=='TN_J0121':
		vmin, vmax = vmin/4., vmax/4.

	elif source=='TNJ0205':
		vmin, vmax = vmin/25., vmax/25.

	elif source=='4C19':
		vmin, vmax = vmin/5, vmax/5.

	elif source=='TN_J1338':
		vmin, vmax = vmin/10., vmax

	irac_fig = ax.imshow(irac_data, transform=ax.get_transform(irac_wcs), origin='lower', 
		interpolation='nearest', vmin=vmin, vmax=vmax, cmap='gist_gray_r')

	left, bottom, width, height = ax.get_position().bounds
	cax = fig.add_axes([ left*20., 0.11, width*0.04, height ])
	cb = pl.colorbar(irac_fig, orientation = 'vertical', cax=cax)
	cb.set_label(r'SB / MJy sr$^{-1}$',rotation=90, fontsize=12)

	ra = ax.coords[0]
	ra.set_major_formatter('hh:mm:ss.s')
	
	ax.text(65, 30, '10 kpc', color='red', fontsize=10, 
		bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5)
	ax.plot([66, 66+dl], [29., 29.], c='red', lw='2', zorder=10.)

	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize('14')

	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize('14')

	return pl.savefig(save_path+source+'_irac_CI_VLA_img.png')

def muse_lya_irac_CI( Lya_subcube, Lya_img, muse_lya_rms, lam1, lam2,
	irac_path, irac_img, irac_rms, CI_path, CI_moment0, CI_rms, 
	radio_hotspots, dl, save_path ):
	"""
	Overlay [CI] contours over Spitzer/IRAC imaging

	Parameters 
	----------
	Lya_subcube : str
		Continuum-subtracted Ly-alpha subcube

	Lya_img : str
		Shorthand name of Ly-alpha subcube

	muse_lya_rms : float
		Minimum threshold of Ly-alpha contours

	lam1 : float
		Lower limit of narrow-band of Ly-alpha image

	lam2 : float
		Upper limit of narrow-band of Ly-alpha image

	irac_path : str
		Path of IRAC image

	irac_img : str
		Filename of IRAC image

	irac_rms : float
		Minimum threshold of IRAC rms

	CI_path : str 
		Path for ALMA [CI] datacubes

	CI_moment0 : str
		Filename of [CI] moment-0 map

	CI_rms : float
		Minimum threshold of [CI] contours

	radio_hotspots : 1d array
		Pixel co-ordinates of radio hotspots in MUSE image

	dl : float
		Length of distance scale bar  

	save_path : str
		Path for saved output

	Returns 
	-------
	Ly-alpha narrow-band image with [CI] and Spitzer/IRAC contours overlaid 

	"""
	CI_fn = CI_contours(CI_path, CI_moment0, CI_rms)

	CI_data, CI_wcs, contours 	= CI_fn[0], CI_fn[1], CI_fn[2]

	fig 		= pl.figure(figsize=(7,5))
	
	ax1 = fig.add_axes([0.02, 0.1, 0.95, 0.85], projection=CI_wcs)

	ax1.contour(CI_data, levels=contours, colors='blue', label='[CI](1-0)', zorder=5)

	muse_Lya = mpdo.Cube(save_path+Lya_subcube, ext=1)

	# collapse along Lya profile axis to form image
	spec = muse_Lya.sum(axis=(1,2))

	p1,p2 = spec.wave.pixel([lam1,lam2], nearest=True)

	muse_img = muse_Lya[p1:p2+1, :, :].sum(axis=0)
	muse_img_arr = muse_img.data[:,:]
	x = len(muse_img_arr)
	y = len(muse_img_arr[0])
	delta_lam = abs(lam2-lam1)		#bandwidth of image

	muse_img_arr = [ [muse_img_arr[i][j]/delta_lam/0.04e3 for j in range(y)] for i in range(x) ]

	muse_img.write(save_path+Lya_img+'_'+`int(lam1)`+'_'+`int(lam2)`+'.fits')		#save to create new WCS		

	muse_hdu = fits.open(save_path+Lya_img+'_'+`int(lam1)`+'_'+`int(lam2)`+'.fits')	#open saved narrow-band image
	muse_hdr = muse_hdu[1].header
	muse_wcs = WCS(muse_hdr).celestial

	pix = list(chain(*muse_img_arr))
	pix_rms = np.sqrt(np.mean(np.square(pix)))
	pix_med = np.median(pix)
	vmax = 5*(pix_med + pix_rms) 
	vmin = 0.2*(pix_med - pix_rms) 

	muse_fig = ax1.imshow(muse_img_arr, transform=ax1.get_transform(muse_wcs), origin='lower', interpolation='nearest', 
		cmap='gist_gray_r', vmin=vmin, vmax=vmax)

	lya_fn = muse_lya_contours( muse_lya_rms, save_path+Lya_img+'_'+`int(lam1)`+'_'+`int(lam2)`+'.fits' )
	lya_data, lya_wcs, contours = lya_fn[0], lya_fn[1], lya_fn[2]

	ax1.contour( lya_data, levels=contours, colors='grey',
	transform=ax1.get_transform(muse_wcs) ) 

	for (h1,h2) in radio_hotspots:
		ax1.scatter(h1, h2, marker='X', transform=ax1.get_transform(muse_wcs), facecolor='green', s=100, zorder=10, edgecolor='black')

	left, bottom, width, height = ax1.get_position().bounds
	cax = fig.add_axes([ left*42., 0.11, width*0.04, height ])
	cb = pl.colorbar(muse_fig, orientation = 'vertical', cax=cax)
	cb.set_label(r'SB / 10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$', rotation=90, fontsize=12)

	irac_fn = irac_contours( irac_path, irac_img, irac_rms, 6 )

	irac_data, irac_wcs, contours = irac_fn[0], irac_fn[1], irac_fn[2]

	ax1.contour( irac_data, levels=contours, colors='red',
	transform=ax1.get_transform(irac_wcs) )

	if Lya_img == '4C03_Lya_img':
		l1,l2 = 25,65
		ax1.set_xlim(l1,l2)
		ax1.set_ylim(l1,l2)

	else:
		l1,l2 = 30,70
		ax1.set_xlim(l1,l2)
		ax1.set_ylim(l1,l2)

	ax1.text( l2-8, l1+5,'10 kpc', color='red', fontsize=10, 
		bbox={'facecolor':'white', 'alpha':0.7, 'pad':10}, zorder=5, transform=ax1.get_transform(CI_wcs))
	sp = l2-7.5
	ax1.plot( [sp, sp+dl], [l1+4,l1+4], c='red', lw='2', zorder=10.)

	ax1.set_xlabel(r'$\alpha$ (J2000)', fontsize=14)
	ax1.set_ylabel(r'$\delta$ (J2000)', fontsize=14)

	ra = ax1.coords[0]
	ra.set_major_formatter('hh:mm:ss.s')

	return pl.savefig(save_path+irac_img[:-5]+'_CI.png')

def muse_lya_vla_CI( Lya_subcube, vla_path, vla_img, 
	CI_path, CI_moment0, Lya_img, lam1, lam2, save_path ):
	"""
	Visualise MUSE Lyman-alpha image with ALMA [CI]
	and VLA C-band contours 

	Parameters 
	----------
	muse_path : str
		Path for MUSE datacube

	muse_cube : str 
		Filename for MUSE datacube

	vla_path : str
		Path for VLA image
	
	vla_img : str
		Filename for VLA image

	CI_path : str
		Path for [CI] moment-0 map

	CI_moment0 : str
		Filename for [CI] moment-0 maps

	Lya_img : str
		Filename for Ly-alpha image

	lam1 : str
		Lower wavelength limit of Lya narrow band image

	lam2 : str
		Upper wavelength limit of Lya narrow band image

	save_path : str
		Path for saved output

	Return
	------
	Multiwavelength narrow-band image with MUSE Lyman-alpha image 
	with ALMA [CI] line and VLA C-band contours 

	"""
	fig = pl.figure(figsize=(10,6))

	# mpl.rcParams['xtick.direction'] = 'in'
	# mpl.rcParams['ytick.direction'] = 'in'

	# mpl.rcParams['xtick.color'] = 'white'
	# mpl.rcParams['ytick.color'] = 'white'

	muse_hdu = fits.open(save_path+Lya_fits)	#open saved narrow-band image
	muse_hdr = muse_hdu[1].header
	muse_wcs = WCS(muse_hdr).celestial

	ax2 = fig.add_subplot(122, projection=muse_wcs)
	muse_cmap = ax2.imshow(muse_img_arr, origin='lower', interpolation='nearest', 
		cmap='gist_gray_r', vmin=-5, vmax=100)

	left, bottom, width, height = ax2.get_position().bounds
	cax = fig.add_axes([ 1.6*left, 0.1, width*0.04, height ])
	cb = pl.colorbar(muse_cmap, orientation = 'vertical', cax=cax)
	cb.set_label(r'SB / 10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ pix$^{-1}$', rotation=90, fontsize=16)
	cb.ax.tick_params(labelsize=18)

	ax2.set_xlim(10,80)
	ax2.set_ylim(10,100)

	vla_fn 		= VLA_contours(vla_path, vla_img)

	vla_data, vla_wcs, vla_contours = vla_fn[0],  vla_fn[1], vla_fn[2]

	ax2.contour(vla_data, levels=vla_contours, colors='green', 
		transform=ax2.get_transform(vla_wcs), label='VLA C-band')

	CI_fn = CI_contours(CI_path, CI_moment0, CI_rms)

	CI_data, CI_wcs, ci_contours 	= CI_fn[0], CI_fn[1], CI_fn[2]

	ax2.contour(CI_data, levels=ci_contours, colors='blue', 
		transform=ax2.get_transform(CI_wcs), label='[CI](1-0)')

	ax1 = fig.add_subplot(121, projection=muse_wcs)

	ax1.imshow(muse_img_arr, origin='lower', interpolation='nearest', 
		cmap='gist_gray_r', vmin=-5, vmax=100.)

	muse_fn = muse_lya_contours( 100., 8, save_path+Lya_fits)

	muse_data, muse_contours = muse_fn[0], muse_fn[2]

	ax1.contour(muse_data, levels=muse_contours, colors='red'
		, label=r'MUSE Ly$\alpha$')

	ax1.contour(CI_data, levels=ci_contours, colors='blue', 
		transform=ax1.get_transform(CI_wcs), label='[CI](1-0)')

	ax1.set_xlim(10,80)
	ax1.set_ylim(10,100)

	ra1 = ax1.coords[1]
	ra1.set_major_formatter('hh:mm:ss.s')

	ra2 = ax2.coords[1]
	ra2.set_ticklabel_visible(0)

	ax1.set_ylabel(r'$\delta$ (J2000)', fontsize=16)
	fig.text(0.45, 0.02, r'$\alpha$ (J2000)', fontsize=16)

	# ax1.set_xticklabel(size=14)
	# ax1.set_yticklabel(size=14)
	# ax2.set_xticklabel(size=14)

	pl.subplots_adjust(wspace=0, left=0.13, right=0.85)	

	return pl.savefig(save_path+'muse_Lya_img_'+`int(lam1)`+'_'+`int(lam2)`+'.png')

def muse_lya_continuum_subtract( muse_path, muse_cube, lam1, lam2, mask_lmin, mask_lmax,
	ra_lim, dec_lim, source, save_path ): 
	"""
	Continuum-subtract MUSE Lya subcube

	Parameters 
	----------
	muse path : str
		Path for MUSE cube

	muse_cube : str
		Filename for MUSE datacube 

	lam1 : float
		Lower wavelength limit of subcube

	lam2 : float
		Upper wavelength limit of sucbube 

	mask_lmin : float
		Lower wavelength limit of spectral region mask

	mask_lmax : float
		Upper wavelength limit of spectral region mask

	ra_lim : float
		Right ascension pixel of subcube 

	dec_lim : float
		Declination pixel range of subcube
	
	source : str
		Name of source

	save_path : str
		Path for saved output

	Returns 
	-------
	Continuum subtracted Ly-alpha subcube

	"""
	# MUSE Lya image
	# open original cube
	start_time = time.time()

	muse_file = muse_path+muse_cube
	muse_hdu = fits.open(muse_file)

	muse_mpdaf_cube = mpdo.Cube(muse_file, mmap=True)
	rg = muse_mpdaf_cube[:, dec_lim[0]:dec_lim[1], ra_lim[0]:ra_lim[1]]	# select F.O.V.

	# identify Ly-alpha line in spectrum
	spec = rg.sum(axis=(1,2))

	# select spectral range and smaller F.O.V. (in pixels)
	p1,p2 = spec.wave.pixel([lam1,lam2], nearest=True)

	Lya_emi = rg[ p1:p2+1, :, : ]				# select spectral region

	Lya_emi.write(save_path+source+'_Lya_subcube.fits')

	print 'Initialising empty cube onto which we write the continuum solution...'

	cont 			= Lya_emi.clone(data_init = np.empty, var_init = np.empty) 	#empty cube	
	Lya_emi_copy 	= Lya_emi.copy()												#copy of cube

	print 'Masking copy of sub-cube...'

	for sp in mpdo.iter_spe(Lya_emi_copy):		#mask Lya line emission in each spaxel
		sp.mask_region(lmin=mask_lmin,lmax=mask_lmax)

	print 'Calculating continuum subtraction solution...'

	for sp,co in zip(mpdo.iter_spe(Lya_emi_copy),mpdo.iter_spe(cont)):
		co[:] = sp.poly_spec(0)

	print 'Subtracting continuum...'

	# continuum subtracted cube
	Lya_cs = Lya_emi - cont	
	Lya_cs.write(save_path+source+'_Lya_subcube_cs.fits')

	elapsed = (time.time() - start_time)/60.
	print "Process complete. Total build time: %f mins" % elapsed

def muse_lya_astro_correct( muse_std, gaia_std, Lya_subcube, save_path):
	"""
	MUSE astrometry correct

	Parameters 
	----------
	muse_std : tuple
		ra, dec of field star in MUSE frame

	gaia_std : tuple 
		ra, dec of field star in GAIA frame

	Lya_subcube : str
		Name of Ly-alpha subcube

	save_path : str
		Path for saved output

	Returns 
	-------
	Astrometry-corrected Ly-alpha subcube

	"""
	hdu = fits.open(save_path+Lya_subcube)
	hdr = hdu[0].header

	ra_offset = muse_std[0] - gaia_std[0]
	dec_offset = muse_std[1] - gaia_std[1]

	hdr['RA'] = hdr['RA'] + ra_offset
	hdr['DEC'] = hdr['DEC'] + dec_offset

	hdu.writeto(save_path+Lya_subcube[:-5]+'_astrocorr.fits', overwrite=1)

def gauss1( x, amp, wid, g_cen, cont ):
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
	gauss = (amp/np.sqrt(2.*np.pi)/wid) * np.exp(-(x-g_cen)**2 /(2*wid**2))
	return gauss + cont

def gauss2( x, a, wid, g_cen, cont ):
	"""
	Gaussian function with continuum
	Peak flux is a fit parameter

	Parameters 
	----------
	x : array
		Wavelength axis

	a : float
		Peak flux

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
	gauss = a*np.exp(-(x-g_cen)**2 /(2*wid**2))
	return gauss + cont

def str_line( x, grad, cut ):
	"""
	Function for 1st-order polynomial 

	Parameters 
	----------
	x : float
		Abcissa axis

	grad : float
		Gradient of the 1st-order polynomial

	cut : float
		Ordinate intercept

	Returns
	-------
	Straight-line function : 1d array	

	"""
	return grad*x + cut