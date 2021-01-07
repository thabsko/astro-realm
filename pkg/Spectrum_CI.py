"""
S.N. Kolwa
ESO (2019) 

"""

from astropy.io import fits
import numpy as np

import matplotlib.pyplot as pl
from matplotlib.ticker import FormatStrFormatter

from lmfit import *
import lmfit.models as lm

from Gaussian import * 
from Image_CI import *

import pickle

from itertools import chain


class Spectrum_CI:

	def __init__( self, input_dir=None, output_dir=None  ):
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
	
	def make_spectrum( self, CI_path, CI_moment0, 
		CI_region, source, p, z, z_err, freq_obs_mt0 ):
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

		print('{:.2f} x {:.2f} arcsec^2'.format(bmaj*3600, bmin*3600))
	
		k = -1 
		for spec in CI_region:
			k += 1
			# 10 km/s binned 1D spectrum
			alma_spec = np.genfromtxt(self.input_dir+source+'_spec_'+spec+'_freq.txt')
	
			freq = alma_spec[:,0]	# GHz
			flux = alma_spec[:,1]	# Jy			
			flux = [ alma_spec[:,1][i]*1.e3 for i in range(len(flux)) ] # mJy

			# Draw spectrum
			fig = pl.figure(figsize=(7,5))
			fig.add_axes([0.15, 0.1, 0.8, 0.8])
			ax = pl.gca()

			# For [CI] detections
			if spec != 'host' or source=='TNJ0121' or source=='4C03' or source=='4C19': 
				print( '-'*len('   '+spec+' '+source+'   '))
				print('   '+spec+' '+source+'   ')
				print( '-'*len('   '+spec+' '+source+'   '))

				pars = Parameters()
				if source=='MRC0943' and p[k][0]=='Loki': #Two component fit
					
					pars.add_many( 
						('g_cen1', p[k][1], True, 0.),
						('a1', p[k][2], True, 0.),	
						('wid1', p[k][3], True, 0.), 
						('g_cen2', p[k][4], True, 0.),
						('a2', p[k][5], True, 0.),	
						('wid2', p[k][6], True, 0.), 
						('cont', p[k][7], True ))
					
					mod 	= lm.Model(Gaussian.dgauss) 
					fit 	= mod.fit(flux, pars, x=freq)
					print( fit.fit_report() )
						
					res = fit.params

					
					sigma, sigma_err 	= (res['wid1'].value, res['wid2']), (res['wid2'].stderr, res['wid1'].stderr)
					freq_o, freq_o_err 	= (res['g_cen1'].value, res['g_cen2'].value), (res['g_cen1'].stderr, res['g_cen2'].stderr)
					flux_peak, flux_peak_err = (res['a1'].value, res['a2'].value), (res['a1'].stderr, res['a2'].stderr)	 # peak flux in mJy	
	
					# Physical parameters for each emission component
					for s,serr,f,ferr,fpeak,fpeakerr in zip(sigma,sigma_err,freq_o,freq_o_err,flux_peak, flux_peak_err): 
						# Line-width in km/s
						sigma_kms = (s/freq_em)*self.c
						sigma_kms_err = sigma_kms * (serr/s)
		
						# FWHM in km/s
						fwhm_kms, fwhm_kms_err = self.convert_fwhm_kms( s, serr, f, ferr )
		
						print( "Sigma (km/s) = %.3f +/- %.3f" %(sigma_kms, sigma_kms_err))
						print( "FWHM (km/s) = %.0f +/- %.0f" %(fwhm_kms, fwhm_kms_err))
		
						# Integrated Flux in mJy km/s
						SdV 		= fpeak * fwhm_kms 				
						SdV_err 	= SdV * (fpeakerr/fpeak)
						print( "Flux peak (mJy) = %.3f +/- %.3f" %(fpeak, fpeakerr) )
						print( "Flux integrated (mJy km/s) = %.0f +/- %.0f" %(SdV, SdV_err) )
		
						# Inferred H_2 mass
						M_H2 = Image_CI.get_mass(self, z, z_err, SdV, SdV_err, f, ferr)	

						
		
						# Velocity shifts
						freq_sys = freq_em/(1.+z)
		
						v_sys = self.c*(1. - freq_sys/freq_em)
						v_obs = self.c*(1. - f/freq_em)
		
						vel_offset = v_obs - v_sys
						vel_offset_err = vel_offset * (ferr/f)
		
						print( "Velocity shift (km/s) = %.3f +/- %.3f" %( vel_offset, vel_offset_err ) )

				# Single component fit
				else:
					pars = Parameters()

					pars.add_many( 
						('g_cen', p[k][1], True, 0.),
						('a', p[k][2], True, 0.),	
						('wid', p[k][3], True, 0.),  	
						('cont', p[k][4], True ))
					
					mod 	= lm.Model(Gaussian.gauss) 
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
					SdV_err 	= SdV * (flux_peak_err/flux_peak)
		
					print( "Flux peak (mJy) = %.3f +/- %.3f" %(flux_peak, flux_peak_err) )
					print( "Flux integrated (mJy km/s) = %.0f +/- %.0f" %(SdV, SdV_err) )
		
					# Inferred H_2 mass
					M_H2 = Image_CI.get_mass(self, z, z_err, SdV, SdV_err, freq_o, freq_o_err)	
		
					# Velocity shifts
					freq_sys = freq_em/(1.+z)
		
					v_sys = self.c*(1. - freq_sys/freq_em)
					v_obs = self.c*(1. - freq_o/freq_em)
		
					vel_offset = v_obs - v_sys
					vel_offset_err = vel_offset * (freq_o_err/freq_o)
		
					print( "Velocity shift (km/s) = %.3f +/- %.3f" %( vel_offset, vel_offset_err ) )
	
				# Frequency range of moment-0 map
				v_obs_mt0 = ( self.c*(1. - freq_obs_mt0[0]/freq_em) - v_sys, 
					self.c*(1. - freq_obs_mt0[1]/freq_em) - v_sys )
	
				print('Moment-0 map velocity range: %.2f to %.2f km/s' %(v_obs_mt0[1], v_obs_mt0[0]) )
	
				freq_ax = np.linspace( freq.min(), freq.max(), num=len(freq))

				if source=='MRC0943' and p[k][0] == 'Loki':
					ax.plot(freq_ax, Gaussian.dgauss(freq_ax, res['a1'], res['wid1'], 
					res['g_cen1'], res['a2'], res['wid2'], res['g_cen2'], res['cont']), c='red')

					fit_params = [ res['a1'], res['a1'].stderr,  res['wid1'], res['wid1'].stderr,
					res['g_cen1'], res['g_cen1'].stderr, res['a2'], res['a2'].stderr,  res['wid2'], 
					res['wid2'].stderr, res['g_cen2'], res['g_cen2'].stderr, res['cont'], res['cont'].stderr ]

					np.savetxt(self.output_dir+spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
						header = 'a1   a1_err   wid1   wid1_err   g_cen1   g_cen2_err  a2   a2_err   wid2   wid2_err   g_cen2   g_cen2_err 	  cont   cont_err')

				else: 
					ax.plot(freq_ax, Gaussian.gauss(freq_ax, res['a'], res['wid'], 
					res['g_cen'], res['cont']), c='red')
	
					fit_params = [ res['a'], res['a'].stderr,  res['wid'], res['wid'].stderr,
					res['g_cen'], res['g_cen'].stderr, res['cont'], res['cont'].stderr ]
	
					np.savetxt(self.output_dir+spec+'_fit_params.txt', (fit_params,), fmt='%.4f', 
						header = 'a   a_err   wid   wid_err   g_cen   g_cen_err  cont   cont_err')

			# No [CI] detections
			else:
				flux = flux
			
			ax.plot( freq, flux, c='k', drawstyle='steps-mid' )

			if source !='4C03' or 'TNJ0121':
				alma_spec_50 = np.genfromtxt(self.input_dir+source+'_spec_host_50kms_freq.txt')

				freq_50 = alma_spec_50[:,0]	
				flux_50 = alma_spec_50[:,1]
				flux_50 = [ flux_50[i]*1.e3 for i in range(len(flux_50)) ]

				ax.plot( freq_50, flux_50, c='#0a78d1', drawstyle='steps-mid' )

			else: 
				print("Host Galaxy has [CI]1-0 detection!")

			pl.savefig(self.output_dir+'CI_spec_'+spec+'.png')
		
			# Pickle figure and save to re-open again
			pickle.dump( fig, open( self.output_dir+'CI_spec_'+spec+'.pickle', 'wb' ) )

			# Get the axes to put the data into subplots
			data1, data2 = [],[]

		for spec in CI_region:
			mpl_ax = pickle.load(open(self.output_dir+'CI_spec_'+spec+'.pickle', 'rb'))

			data1.append(mpl_ax.axes[0].lines[0].get_data())	# model or finely smoothed spectrum for non-detection
			data2.append(mpl_ax.axes[0].lines[1].get_data())	# data 	or widely smoothed spectrum for non-detection

		no_subplots = len(data1)
		
		fig, ax = pl.subplots(no_subplots, 1, figsize=(6, 3*no_subplots), sharex=True, constrained_layout=True)
		pl.subplots_adjust(hspace=0, wspace=0.01) 

		# Top frequency axis
		ax_top = ax[0].twiny()
		freq_top = data1[0][0]
		ax_top.plot( freq_top, data1[0][1], c='None')
		ax_top.set_xlabel('Frequency (GHz)', fontsize=14)
		top_xticks = np.arange(min(freq_top), max(freq_top), 0.25)
		ax_top.set_xticks( top_xticks )
		ax_top.invert_xaxis()
		ax_top.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

		# Velocity axes 
		v_radio1 = [ self.freq_to_vel(data1[1][0][i], freq_em, 0.) for i in range(len(data1[1][0])) ] #model fit
		v_radio2 = [ self.freq_to_vel(data2[1][0][i], freq_em, 0.) for i in range(len(data2[1][0])) ] #data 
		v_radio3 = [ self.freq_to_vel(data2[0][0][i], freq_em, 0.) for i in range(len(data2[0][0])) ] #widely smoothed data

		# Frequency at the systemic redshift (from HeII)
		freq_sys = freq_em/(1.+z)				
		vel0 = self.freq_to_vel(freq_sys, freq_em, 0.)
		
		# Velocity offset axes
		voff1 = [ v_radio1[i] - vel0 for i in range(len(v_radio1)) ]		#model fit
		voff2 = [ v_radio2[i] - vel0 for i in range(len(v_radio2)) ]		#data
		voff_wide = [ v_radio3[i] - vel0 for i in range(len(v_radio3)) ]	#widely smoothed data

		# Global plot parameters
		fs = 14
		dx = 0.05
		dy = 0.92

		# Custom plot parameters per source
		if source == '4C03':
			host_cont = np.genfromtxt(self.output_dir+'host_fit_params.txt')[6]
			ax[0].plot(voff1, data1[0][1], c='red')
			ax[0].plot(voff2, data2[0][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > 240. and voff2[i] < 280.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data1_2_sub = [ data2[0][1][i] for i in indices ]
			ax[0].fill_between( voff2_sub, data1_2_sub, host_cont, interpolate=1, color='yellow' )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')		
			
			SE_cont = np.genfromtxt(self.output_dir+'SE_fit_params.txt')[6]
			ax[1].plot( voff1, data1[1][1], c='red' )
			ax[1].plot( voff2, data2[1][1], c='k', drawstyle='steps-mid' )
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -360 and voff2[i] < -300.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, SE_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) South-east', ha='left', transform=ax[1].transAxes, fontsize=fs, color='k')
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				miny,maxy,dy = -1.0, 1.8, 0.5
				ax.plot([0.,0.], [miny, maxy], c='gray', ls='--')
				ax.set_yticks( np.arange(miny, maxy, dy) )
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.set_ylim([miny+0.1, maxy])

			pl.savefig(self.output_dir+'4C03_CI_spectrums.png', bbox_inches = 'tight',
    		pad_inches = 0.1)

		elif source == '4C04':
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_wide, data2[0][1], c='#0a78d1', drawstyle='steps-mid', lw=2)
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')
			
			NW_cont = np.genfromtxt(self.output_dir+'NW_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -350. and voff2[i] < -270.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-east', ha='left', transform=ax[1].transAxes, fontsize=fs, color='k')
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)
			
			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				miny,maxy,dy = -0.25, 2.5, 0.5
				ax.plot([0.,0.], [miny, maxy], c='gray', ls='--')
				ax.set_yticks( np.arange(miny, maxy, dy) )
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.set_ylim([miny+0.1, maxy])

			pl.savefig(self.output_dir+'4C04_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == '4C19':
			host_cont = np.genfromtxt(self.output_dir+'host_fit_params.txt')[6]
			ax[0].plot(voff1, data1[0][1], c='red')
			ax[0].plot(voff2, data2[0][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -10. and voff2[i] < 30.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data1_2_sub = [ data2[0][1][i] for i in indices ]
			ax[0].fill_between( voff2_sub, data1_2_sub, host_cont, interpolate=1, color='yellow' )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')		

			NE_cont = np.genfromtxt(self.output_dir+'NE_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -120. and voff2[i] < -20.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NE_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North-east', ha='left', transform=ax[1].transAxes, fontsize=fs, color='k')

			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				miny,maxy,dy = -1.5, 2.8, 0.5
				ax.plot([0.,0.], [miny, maxy], c='gray', ls='--')
				ax.set_yticks( np.arange(miny, maxy, dy) )
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.set_ylim([miny+0.1, maxy])
		
			pl.savefig(self.output_dir+'4C19_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'MRC0943':
			fig, ax = pl.subplots(1, 3, figsize=(18, 3), sharex=True, sharey=True, constrained_layout=True)
			pl.subplots_adjust(hspace=0, wspace=0) 
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_wide, data2[0][1], c='#0a78d1', drawstyle='steps-mid', lw=2)
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')
			ax[0].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			SW_cont = np.genfromtxt(self.output_dir+'SW_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -290. and voff2[i] < 190.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, SW_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) Thor/Odin ', ha='left', transform=ax[1].transAxes, fontsize=fs, color='k')
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			Loki_cont = np.genfromtxt(self.output_dir+'Loki_fit_params.txt')[12]
			ax[2].plot(voff1, data1[2][1], c='red')
			ax[2].plot(voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -210. and voff2[i] < 0.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[2].fill_between( voff2_sub, data3_2_sub, Loki_cont, interpolate=1, color='yellow' )
			ax[2].text(dx, dy, '(c) Loki', ha='left', transform=ax[2].transAxes, fontsize=fs, color='k')
			ax[2].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			ax[0].set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs) #ylabel for left-side plots only

			for j in range(3): 	#top axes
				ax_top = ax[j].twiny()
				freq_top = data1[0][0]
				ax_top.plot( freq_top, data1[0][1], c='None')
				ax_top.set_xlabel('Frequency (GHz)', fontsize=14)
				top_xticks = np.arange(min(freq_top), max(freq_top), 0.25)
				ax_top.set_xticks( top_xticks )
				ax_top.invert_xaxis()
				ax_top.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

			for ax in ax:
				miny,maxy,dy = -1., 2.0, 0.5
				ax.plot([0.,0.], [miny, maxy], c='gray', ls='--')
				ax.set_yticks( np.arange(miny, maxy, dy) )
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.set_ylim([miny+0.1, maxy])
		
			pl.savefig(self.output_dir+'MRC0943_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'TNJ0121':
			host_cont = np.genfromtxt(self.output_dir+'host_fit_params.txt')[6]
			ax[0].plot(voff1, data1[0][1], c='red')
			ax[0].plot(voff2, data2[0][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -30. and voff2[i] < 20.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data1_2_sub = [ data2[0][1][i] for i in indices ]
			ax[0].fill_between( voff2_sub, data1_2_sub, host_cont, interpolate=1, color='yellow' )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')
		
			NW_far_cont = np.genfromtxt(self.output_dir+spec+'_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -390. and voff2[i] < -320.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, NW_far_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) South-west', ha='left', transform=ax[1].transAxes, fontsize=fs, color='k')
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			for ax in ax:
				ax.set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs)
				miny,maxy,dy = -1., 2.0, 0.5
				ax.plot([0.,0.], [miny, maxy], c='gray', ls='--')
				ax.set_yticks( np.arange(miny, maxy, dy) )
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.set_ylim([miny+0.1, maxy])
		
			pl.savefig(self.output_dir+'TNJ0121_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'TNJ0205':
			fig, ax = pl.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
			pl.subplots_adjust(hspace=0, wspace=0) 
			ax[0][0].plot(voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0][0].plot(voff_wide, data2[0][1], c='#0a78d1', drawstyle='steps-mid', lw=2 )
			ax[0][0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0][0].transAxes, fontsize=fs, color='k')
		
			NW_cont = np.genfromtxt(self.output_dir+'NW_fit_params.txt')[6]
			ax[0][1].plot(voff1, data1[1][1], c='red')
			ax[0][1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > 330. and voff2[i] < 410.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[0][1].fill_between( voff2_sub, data2_2_sub, NW_cont, interpolate=1, color='yellow' )
			ax[0][1].text(dx, dy, '(b) North-west', ha='left', transform=ax[0][1].transAxes, fontsize=fs, color='k')
		
			SW_cont = np.genfromtxt(self.output_dir+'SW_fit_params.txt')[6]
			ax[1][0].plot(voff1, data1[2][1], c='red')
			ax[1][0].plot(voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > 50. and voff2[i] < 200.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[1][0].fill_between( voff2_sub, data3_2_sub, SW_cont, interpolate=1, color='yellow' )
			ax[1][0].text(dx, dy, '(c) South-west', ha='left', transform=ax[1][0].transAxes, fontsize=fs, color='k')
			ax[1][0].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			SE_cont = np.genfromtxt(self.output_dir+'SE_fit_params.txt')[6]
			ax[1][1].plot(voff1, data1[3][1], c='red')
			ax[1][1].plot(voff2, data2[3][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -380. and voff2[i] < -320.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[3][1][i] for i in indices ]
			ax[1][1].fill_between( voff2_sub, data3_2_sub, SE_cont, interpolate=1, color='yellow' )
			ax[1][1].text(dx, dy, '(c) South-east', ha='left', transform=ax[1][1].transAxes, fontsize=fs, color='k')
			ax[1][1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			for h in range(2):
				ax[h][0].set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs) #ylabel for left-side plots only

			for j in range(2): 	#top axes
				ax_top = ax[0][j].twiny()
				freq_top = data1[0][0]
				ax_top.plot( freq_top, data1[0][1], c='None')
				ax_top.set_xlabel('Frequency (GHz)', fontsize=14)
				top_xticks = np.arange(min(freq_top), max(freq_top), 0.25)
				ax_top.set_xticks( top_xticks )
				ax_top.invert_xaxis()
				ax_top.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

			ax =  list(chain(*ax))	#unify nested array
			for ax in ax:
				miny,maxy,dy = -1., 2.5, 0.5
				ax.plot([0.,0.], [miny, maxy], c='gray', ls='--')
				ax.set_yticks( np.arange(miny, maxy, dy) )
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.set_ylim([miny+0.1, maxy])
				
			pl.savefig(self.output_dir+'TNJ0205_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)

		elif source == 'TNJ1338':
			fig, ax = pl.subplots(1, 3, figsize=(18, 3), sharex=True, sharey=True, constrained_layout=True)
			pl.subplots_adjust(hspace=0, wspace=0) 
			ax[0].plot( voff2, data1[0][1], c='k', drawstyle='steps-mid' )
			ax[0].plot( voff_wide, data2[0][1], c='#0a78d1', drawstyle='steps-mid' )
			ax[0].text(dx, dy, '(a) Host Galaxy', ha='left', transform=ax[0].transAxes, fontsize=fs, color='k')
			ax[0].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			N_cont = np.genfromtxt(self.output_dir+'N_fit_params.txt')[6]
			ax[1].plot(voff1, data1[1][1], c='red')
			ax[1].plot(voff2, data2[1][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -800. and voff2[i] < -720.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data2_2_sub = [ data2[1][1][i] for i in indices ]
			ax[1].fill_between( voff2_sub, data2_2_sub, N_cont, interpolate=1, color='yellow' )
			ax[1].text(dx, dy, '(b) North', ha='left', transform=ax[1].transAxes, fontsize=fs, color='k')
			ax[1].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			SW_cont = np.genfromtxt(self.output_dir+'SW_fit_params.txt')[6]
			ax[2].plot(voff1, data1[2][1], c='red')
			ax[2].plot(voff2, data2[2][1], c='k', drawstyle='steps-mid')
			indices = [ i for i in range(len(voff2)) if (voff2[i] > -430. and voff2[i] < -250.)  ]
			voff2_sub = [ voff2[i] for i in indices ]
			data3_2_sub = [ data2[2][1][i] for i in indices ]
			ax[2].fill_between( voff2_sub, data3_2_sub, SW_cont, interpolate=1, color='yellow' )
			ax[2].text(dx, dy, '(c) South-west', ha='left', transform=ax[2].transAxes, fontsize=fs, color='k')
			ax[2].set_xlabel(r'$\Delta v$ (km s$^{-1}$)', fontsize=fs)

			ax[0].set_ylabel(r'S$_\nu$ (mJy)', fontsize=fs) #ylabel for left-side plots only

			for j in range(3): 	#top axes
				ax_top = ax[j].twiny()
				freq_top = data1[0][0]
				ax_top.plot( freq_top, data1[0][1], c='None')
				ax_top.set_xlabel('Frequency (GHz)', fontsize=14)
				top_xticks = np.arange(min(freq_top), max(freq_top), 0.25)
				ax_top.set_xticks( top_xticks )
				ax_top.invert_xaxis()
				ax_top.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

			for ax in ax:
				miny,maxy,dy = -1., 1.2, 0.5
				ax.plot([0.,0.], [miny, maxy], c='gray', ls='--')
				ax.set_yticks( np.arange(miny, maxy, dy) )
				ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
				ax.set_ylim([miny+0.1, maxy])
		
			pl.savefig(self.output_dir+'TNJ1338_CI_spectrums.png', bbox_inches = 'tight',
				pad_inches = 0.1)