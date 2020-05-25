"""
S.N. Kolwa
ESO (2019) 

"""

from astropy.io import fits
import numpy as np

import matplotlib.pyplot as pl
from lmfit import *
import lmfit.models as lm

from Gaussians import * 
from Image_CI import *

import pickle


class Spectrum_CI(object):

	def __init__( self, input_dir, output_dir  ):
		"""
		input_dir : Directory of input files

		output_dir : Directory of output files

		c : light-speed

		"""

		self.input_dir = input_dir	
		self.output_dir = output_dir

		self.c = 2.9979245800e5 	# in km/s

	def freq_to_vel( self, freq_obs, freq_em, z ):
		"""
		Convert an observed frequency to a velocity 
		in the observer-frame at redshift
	
		Parameters 
		----------
		freq_obs : Observed frequency
	
		freq_em : Emitted frequency
	
		z : Observer Redshift
	
		Returns 
		-------
		Velocity (km/s)
		
		"""
		v = self.c*(1. - freq_obs/freq_em/(1.+z) )
		return v
	
	def convert_fwhm_kms( self, sigma, sigma_err, 
		freq_o, freq_o_err ):	
		"""
		Convert velocity dispersion (km/s) to FWHM (km/s)

		Parameters 
		----------
		sigma : Velocity dispersion 

		sigma_err : Velocity dispersion error

		freq_o : Observed frequency

		freq_o_err : Observed frequency error

		Returns 
		-------
		FWHM and FWHM error (km/s) : 1D array
		
		"""
		freq_em = 492.161		#rest frequency of [CI](1-0) in GHz

		fwhm 		= 2.*np.sqrt(2*np.log(2))*sigma 
		fwhm_err 	= (sigma_err/sigma)*fwhm
		
		fwhm_kms	= (fwhm/freq_em)*self.c
		fwhm_kms_err = fwhm_kms*(fwhm_err/fwhm) 
		
		return [fwhm_kms, fwhm_kms_err]
	
	def make_spectrum( self, CI_path, CI_moment0, CI_region, source, p, z,
	 z_err, freq_obs_mt0 ):
		"""
		Show [CI] line spectra and line-fits
	
		Parameters 
		----------
		CI_path : Path of [CI] spectrum 
	
		CI_moment0 : Filename of [CI] moment-0 map

		CI_region : [CI] regions (array)

		source : Short-hand source name

		p : Initial parameters for line-fit

		z : Redshift

		z_err : Redshift error

		freq_obs_mto : Observed frequency range of [CI] moment-0 map
	
		Returns 
		-------
		[CI] spectra : image
		
		"""
		self.CI_path = CI_path
		self.CI_moment0 = CI_moment0
		self.CI_region = CI_region
		self.source = source
		self.p = p
		self.z = z
		self.z_err = z_err
		self.freq_obs_mt0 = freq_obs_mt0

		freq_em = 492.161		#rest frequency of [CI](1-0) in GHz

		moment0 = fits.open(self.CI_path+CI_moment0)
		hdr =  moment0[0].header
	
		bmaj, bmin, bpa = hdr['bmaj'], hdr['bmin'], hdr['bpa']  # bmaj, bmin in degrees
	
		for spec in CI_region:
			# 20 km/s binned 1D spectrum
			alma_spec = np.genfromtxt('../../'+source+'_files/in/'+source+'_spec_'+spec+'_freq.txt')
	
			freq = alma_spec[:,0]	# GHz
			flux = alma_spec[:,1]	# Jy			
			flux = [ alma_spec[:,1][i]*1.e3 for i in range(len(flux)) ] # mJy


			# Draw spectrum
			fig = pl.figure(figsize=(7,5))
			fig.add_axes([0.15, 0.1, 0.8, 0.8])
			ax = pl.gca()

			# For [CI] detections
			if spec != 'host' or source == 'TNJ0121': 
				print( '-'*len('   '+spec+' '+source+'   '))
				print('   '+spec+' '+source+'   ')
				print( '-'*len('   '+spec+' '+source+'   '))

				# Initialise fit parameters
				pars = Parameters()
				pars.add_many( 
					('g_cen', p[0], True, 0.),
					('a', p[1], True, 0.),	
					('wid', p[2], True, 0.),  	
					('cont', p[3], True ))
				
				# Fit [CI] line
				mod 	= lm.Model(Gaussians.gauss_peak) 
				fit 	= mod.fit(flux, pars, x=freq)

				print( fit.fit_report() )
			
				res = fit.params

				# Line-width in km/s
				sigma, sigma_err 	= res['wid'].value, res['wid'].stderr
				freq_o, freq_o_err 	= res['g_cen'].value, res['g_cen'].stderr
	
				sigma_kms = (sigma/freq_em)*self.c
				sigma_kms_err = sigma_kms*np.sqrt ( (sigma_err/sigma)**2 + (freq_o_err/freq_o)**2 )
	
				# FWHM in km/s
				fwhm_kms, fwhm_kms_err = self.convert_fwhm_kms( sigma, sigma_err, freq_o, freq_o_err )
	
				print( "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err))
				print( "FWHM (km/s) = %.0f +/- %.0f" %(fwhm_kms, fwhm_kms_err))
	
				flux_peak, flux_peak_err = res['a'].value, res['a'].stderr	 # peak flux in mJy	
	
				# Integrated Flux in mJy km/s
				SdV 		= flux_peak * fwhm_kms 				
				SdV_err 	= SdV * np.sqrt( (flux_peak_err/flux_peak)**2 + (fwhm_kms_err/fwhm_kms)**2 )
	
				print( "Flux peak (mJy) = %.3f +/- %.3f" %(flux_peak, flux_peak_err) )
				print( "Flux integrated (mJy km/s) = %.0f +/- %.0f" %(SdV, SdV_err) )
	
				# Inferred H_2 mass
				M_H2 = Image_CI.get_mass(self, z, z_err, SdV, SdV_err, freq_o, freq_o_err)	
	
				# Velocity shifts
				freq_sys = freq_em/(1.+z)
	
				v_sys = self.c*(1. - freq_sys/freq_em)
				v_obs = self.c*(1. - freq_o/freq_em)
	
				vel_offset = v_obs - v_sys
				vel_offset_err = vel_offset*( (freq_o_err/freq_o)**2 + (z_err/z)**2 )
	
				print( "Velocity shift (km/s) = %.3f +/- %.3f" %( vel_offset, vel_offset_err ) )
	
				# Frequency range of moment-0 map
				v_obs_mt0 = ( self.c*(1. - freq_obs_mt0[0]/freq_em) - v_sys, 
					self.c*(1. - freq_obs_mt0[1]/freq_em) - v_sys )
	
				print('Moment-0 map velocity range: %.2f to %.2f km/s' %(v_obs_mt0[1], v_obs_mt0[0]) )
	
				freq_ax = np.linspace( freq.min(), freq.max(), num=100)
				ax.plot(freq_ax, Gaussians.gauss_peak(freq_ax, res['a'], res['wid'], 
					res['g_cen'], res['cont']), c='red')
	
				fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
					res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]
	
				np.savetxt(self.output_dir+spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
				header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

			# No [CI] detections
			else:
				flux = flux
			
			ax.plot( freq, flux, c='k', drawstyle='steps-mid' )
			# If a source has a line detection at the host, then...
			if source != 'TNJ0121':
				alma_spec_80 = np.genfromtxt('../../'+source+'_files/in/'+source+'_spec_host_80kms_freq.txt')

				freq_80 = alma_spec_80[:,0]	
				flux_80 = alma_spec_80[:,1]
				flux_80 = [ flux_80[i]*1.e3 for i in range(len(flux_80)) ]

				ax.plot( freq_80, flux_80, c='#0a78d1', drawstyle='steps-mid' )
			else: 
				"No need show 80 km/s binned spectrum."
		
			# Pickle figure and save to re-open again
			pickle.dump( fig, open( self.output_dir+'CI_spec_'+spec+'.pickle', 'wb' ) )

			# Get the axes to put the data into subplots
			data1, data2 = [],[]

		for spec in CI_region:
			mpl_ax = pickle.load(open(self.output_dir+'CI_spec_'+spec+'.pickle', 'rb'))

			data1.append(mpl_ax.axes[0].lines[0].get_data())	# model or 20km/s host
			data2.append(mpl_ax.axes[0].lines[1].get_data())	# data 	or 80km/s host

		no_subplots = len(data1)
		
		fig, ax = pl.subplots(no_subplots, 1, figsize=(6, 3*no_subplots), sharex=True, constrained_layout=True)
		pl.subplots_adjust(hspace=0, wspace=0.01) 

		# Velocity axes 
		v_radio1 = [ self.freq_to_vel(data1[1][0][i], freq_em, 0.) for i in range(len(data1[1][0])) ] #model
		v_radio2 = [ self.freq_to_vel(data2[1][0][i], freq_em, 0.) for i in range(len(data2[1][0])) ] #data 
		v_radio3 = [ self.freq_to_vel(data2[0][0][i], freq_em, 0.) for i in range(len(data2[0][0])) ] #80kms data

		# # Frequency at the systemic redshift (from HeII)
		freq_sys = freq_em/(1.+z)				
		vel0 = self.freq_to_vel(freq_sys, freq_em, 0.)
		
		# # Velocity offset axes
		voff1 = [ v_radio1[i] - vel0 for i in range(len(v_radio1)) ]	#model 
		voff2 = [ v_radio2[i] - vel0 for i in range(len(v_radio2)) ]	#data 
		voff_80 = [ v_radio3[i] - vel0 for i in range(len(v_radio3)) ]	#80kms data

		# Plot parameters
		fs = 14
		dx = 0.05
		dy = 0.92

		# Custom plot parameters for each source
		if source == '4C03':
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_80, data2[0][1], c='#0a78d1', drawstyle='steps-mid', lw=2 )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='red')
			ax[0].set_ylim([ 1.19*min(data1[0][1]), 1.19*max(data1[0][1]) ])	
			
			NW_cont = np.genfromtxt(self.output_dir+'NW_fit_params.txt')[6]
			ax[1].plot( voff1, data1[1][1], c='red' )
			ax[1].plot( voff2, data2[1][1], c='k', drawstyle='steps-mid' )
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 30.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-west', ha='left', transform=ax[1].transAxes, fontsize=fs, color		='magenta')
			ax[1].set_ylim([ 1.19*min(data2[1][1]), 1.19*max(data2[1][1]) ])	
		

			E_cont = np.genfromtxt(self.output_dir+'E_fit_params.txt')[6]
			ax[2].plot( voff1, data1[2][1], c='red' )
			ax[2].plot( voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 50.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[2].fill_between( voff2_sub, data3_2_sub, E_cont, interpolate=1, color='yellow' )
			ax[2].text(dx, dy, '(c) East-south-east', ha='left', transform=ax[2].transAxes, fontsize=fs, 		color='cyan')
			ax[2].set_ylim([ 1.19*min(data2[2][1]), 1.19*max(data2[2][1]) ])	
			
			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				ax.plot([0.,0.], [-0.69, 0.69], c='gray', ls='--')
		
			pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			pl.savefig(self.output_dir+'4C03_CI_spectrums.png', bbox_inches = 'tight',
    		pad_inches = 0.1)

		elif source == '4C04':
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_80, data2[0][1], c='#0a78d1', drawstyle='steps-mid', lw=2)
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 50.)  ]
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='red')
			ax[0].set_ylim([ 1.19*min(data1[0][1]), 1.19*max(data1[0][1]) ])	
			
			NE_cont = np.genfromtxt(self.output_dir+'NE_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -10. and voff2[i] < 70.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NE_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-east', ha='left', transform=ax[1].transAxes, fontsize=fs, color='cyan')
			ax[1].set_ylim([ 1.19*min(data2[1][1]), 1.19*max(data2[1][1]) ])	
			
			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')
		
			pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			pl.savefig(self.output_dir+'4C04_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == '4C19':
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_80, data2[0][1], c='#0a78d1', drawstyle='steps-mid' )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='red')
			ax[0].set_ylim([ 1.19*min(data1[0][1]), 1.19*max(data1[0][1]) ])	
			
			NW_cont = np.genfromtxt(self.output_dir+'NW_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 40.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-north-west', ha='left', transform=ax[1].transAxes, fontsize=fs, color='purple')
			ax[1].set_ylim([ 1.19*min(data2[1][1]), 1.19*max(data2[1][1]) ])	

			SE_cont = np.genfromtxt(self.output_dir+'SE_fit_params.txt')[6]
			ax[2].plot(voff1, data1[2][1], c='red')
			ax[2].plot(voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 40.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[2].fill_between( voff2_sub, data3_2_sub, SE_cont, interpolate=1, color='yellow' )
			ax[2].text(dx, dy, '(c) East-south-east', ha='left', transform=ax[2].transAxes, fontsize=fs, color='orange')
			ax[2].set_ylim([ 1.19*min(data2[2][1]), 1.19*max(data2[2][1]) ])

			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')
		
			pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			pl.savefig(self.output_dir+'4C04_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'MRC0943':
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_80, data2[0][1], c='#0a78d1', drawstyle='steps-mid', lw=2)
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='red')
			ax[0].set_ylim([ 1.19*min(data1[0][1]), 1.19*max(data1[0][1]) ])	
			
			NW_cont = np.genfromtxt(self.output_dir+'NW_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > 20. and voff2[i] < 100.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-west', ha='left', transform=ax[1].transAxes, fontsize=fs, color='orange')
			ax[1].set_ylim([ 1.19*min(data2[1][1]), 1.19*max(data2[1][1]) ])	

			SW_cont = np.genfromtxt(self.output_dir+'SW2_fit_params.txt')[6]
			ax[2].plot(voff1, data1[2][1], c='red')
			ax[2].plot(voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > 20. and voff2[i] < 120.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[2].fill_between( voff2_sub, data3_2_sub, SW_cont, interpolate=1, color='yellow' )
			ax[2].text(dx, dy, '(c) South-west', ha='left', transform=ax[2].transAxes, fontsize=fs, color='magenta')
			ax[2].set_ylim([ 1.19*min(data2[2][1]), 1.19*max(data2[2][1]) ])	

			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')
		
			pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			pl.savefig(self.output_dir+'MRC0943_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'TNJ0121':
			host_cont = np.genfromtxt(self.output_dir+spec+'_fit_params.txt')[6]
			ax[0].plot(voff1, data1[0][1], c='red')
			ax[0].plot(voff2, data2[0][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -60. and voff2[i] < 200.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data1_2_sub = [ data2[0][1][i] for i in indices ]
			ax[0].fill_between( voff2_sub, data1_2_sub, host_cont, interpolate=1, color='yellow' )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='red')
			ax[0].set_ylim([ 1.19*min(data2[0][1]), 1.19*max(data2[0][1]) ])
		
			NW_far_cont = np.genfromtxt(self.output_dir+spec+'_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -60. and voff2[i] < 80.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NW_far_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-west', ha='left', transform=ax[1].transAxes, fontsize=fs, color='#3eff13')
			ax[1].set_ylim([ 1.19*min(data2[1][1]), 1.19*max(data2[1][1]) ])	
		
			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')
		
			pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			pl.savefig(self.output_dir+'TNJ0121_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'TNJ0205':
			ax[0].plot(voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot(voff_80, data2[0][1], c='#0a78d1', drawstyle='steps-mid', lw=2 )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='red')
			ax[0].set_ylim([ 1.19*min(data1[0][1]), 1.19*max(data1[0][1]) ])
		
			E_cont = np.genfromtxt(self.output_dir+'E_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 60.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, E_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) East-north-east', ha='left', transform=ax[1].transAxes, fontsize=fs, color=	'cyan')
			ax[1].set_ylim([ 1.19*min(data2[1][1]), 1.19*max(data2[1][1]) ])	
		
			SE_cont = np.genfromtxt(self.output_dir+'SE_fit_params.txt')[6]
			ax[2].plot(voff1, data1[2][1], c='red')
			ax[2].plot(voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 80.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[2].fill_between( voff2_sub, data3_2_sub, SE_cont, interpolate=1, color='yellow' )
			ax[2].text(dx, dy, '(c) South-east', ha='left', transform=ax[2].transAxes, fontsize=fs, color='orange')
			ax[2].set_ylim([ 1.19*min(data2[2][1]), 1.19*max(data2[2][1]) ])	

			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')
		
			pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			pl.savefig(self.output_dir+'TNJ0205_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'TNJ1338':
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_80, data2[0][1], c='#0a78d1', drawstyle='steps-mid' )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='red')
			ax[0].set_ylim([ 1.19*min(data1[0][1]), 1.19*max(data1[0][1]) ])	

			NW_cont = np.genfromtxt(self.output_dir+'NW_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -50. and voff2[i] < 70.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-north-west', ha='left', transform=ax[1].transAxes, fontsize=fs, color='purple')
			ax[1].set_ylim([ 1.19*min(data2[1][1]), 1.19*max(data2[1][1]) ])

			SW_cont = np.genfromtxt(self.output_dir+'SW_fit_params.txt')[6]
			ax[2].plot(voff1, data1[2][1], c='red')
			ax[2].plot(voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -60. and voff2[i] < 60.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[2].fill_between( voff2_sub, data3_2_sub, SW_cont, interpolate=1, color='yellow' )
			ax[2].text(dx, dy, '(c) South-south-west', ha='left', transform=ax[2].transAxes, fontsize=fs, color='orange')
			ax[2].set_ylim([ 1.19*min(data2[2][1]), 1.19*max(data2[2][1]) ])		

			NE_cont = np.genfromtxt(self.output_dir+'NE_fit_params.txt')[6]
			ax[3].plot(voff1, data1[3][1], c='red')
			ax[3].plot(voff2, data2[3][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -20. and voff2[i] < 80.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data4_2_sub = [ data2[3][1][i] for i in indices ]
			ax[3].fill_between( voff2_sub, data4_2_sub, NE_cont, interpolate=1, color='yellow' )
			ax[3].text(dx, dy, '(d) North-east', ha='left', transform=ax[3].transAxes, fontsize=fs, color='cyan')
			ax[3].set_ylim([ 1.19*min(data2[3][1]), 1.19*max(data2[3][1]) ])	

			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				ax.plot([0.,0.], [-1., 2.], c='gray', ls='--')
		
			pl.xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			pl.savefig(self.output_dir+'TNJ1338_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)
	
