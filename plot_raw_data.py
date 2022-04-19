import sigpyproc as spp
import argparse
from astropy.time import Time
import numpy as np
from tqdm import trange
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

from mpl_toolkits import axes_grid1

import datetime

def plot_data(dataBlock_I, dataBlock_Q, dataBlock_U, dataBlock_V, dataBlock_L, t_init, t_end, plotPrefix, filePrefix, startingSample, percentiles, flux_limits, frequency_limits, figsize,fontsize, bandpass = None, reverse=None, cmap=None, full_stokes = False):

	if full_stokes == True:
		fig, axmain = plt.subplots(5,1,figsize=figsize,sharex=True)
	else:
		fig, axmain = plt.subplots(2,1,figsize=figsize,sharex=True)
	gs = fig.add_gridspec(26, 14)

	axmain[0].set_ylabel("Frequency (MHz)")
	axmain[1].set_ylabel("Frequency (MHz)")

	title_1 = axmain[0].set_title("")
	title_2 = axmain[1].set_title("")

	if full_stokes == True:
		axmain[2].set_ylabel("Frequency (MHz)")
		axmain[3].set_ylabel("Frequency (MHz)")
		axmain[4].set_ylabel("Frequency (MHz)")

		title_3 = axmain[2].set_title("")
		title_4 = axmain[3].set_title("")
		title_5 = axmain[4].set_title("")

	if frequency_limits == None:
		ftop = dataBlock_I.header.ftop
		fbot = dataBlock_I.header.fbottom
	else:
		ftop = frequency_limits[1]
		fbot = frequency_limits[0]


# Color map for the plot
	if cmap == None:
		cmap='Greys'

# Creating x-limits
	if t_init == None:
		time_header = Time(dataBlock_I.header.tstart, format = "mjd")
		print(f"Time from the dataBlock_I.header: {time_header.isot}")
		t_init = datetime.datetime.fromisoformat(time_header.isot)

	if t_end == None:
		time_limit = np.array([t_init+datetime.timedelta(seconds = dataBlock_I.header.tsamp*(dataBlock_I.shape[1])*i) for i in range(2)])
	else:
		time_limit = np.array([t_init, t_end])
	t_end = time_limit[-1]
	time_delta = time_limit[-1] - time_limit[0]

	print(f"Time Limit: {time_limit[0]} to {time_limit[1]}")
# File naming
	filename = f"{filePrefix}_{time_limit[0].strftime('%Y%m%d%H%M%S')}_{time_limit[-1].strftime('%Y%m%d%H%M%S')}.png"


# converting datetime.datetime objects to the correct format for matplotlib to work with
	time_limit = mdates.date2num(time_limit)

# Plotting data - Stokes I
	ind_mplotlib = 0
	if flux_limits:
		vmn, vmx = flux_limits
	else:
		vmx, vmn = np.percentile(dataBlock_I, percentiles)
	ScaleZ=colors.Normalize(vmn,vmx)
	axmainArtist_I = axmain[ind_mplotlib].imshow(dataBlock_I,cmap=cmap, aspect = 'auto', vmin=vmn,vmax=vmx, interpolation ='none', extent =[time_limit[0], time_limit[1], fbot, ftop])
# Setting colorbar
	ax=axmainArtist_I.axes
	fig=ax.figure
	divider = axes_grid1.make_axes_locatable(ax)
	cb = fig.colorbar(axmainArtist_I, extend='both',ax=axmain[ind_mplotlib])
	cb.set_label(r'Intensity',fontsize=fontsize)
	cb.ax.tick_params(labelsize=fontsize)

	ind_mplotlib = ind_mplotlib+1

	if full_stokes == True:
	# Plotting data - Stokes Q
		vmx = 0.5
		vmn = -vmx
		ScaleZ=colors.Normalize(vmn,vmx)
		axmainArtist_Q = axmain[ind_mplotlib].imshow(dataBlock_Q,cmap='Greys', aspect = 'auto', vmin=vmn,vmax=vmx, interpolation= 'none', extent =[time_limit[0], time_limit[1], fbot, ftop])


	# Setting colorbar
		ax=axmainArtist_Q.axes
		fig=ax.figure
		divider = axes_grid1.make_axes_locatable(ax)
		cb = fig.colorbar(axmainArtist_Q, extend='both',ax=axmain[ind_mplotlib])
		cb.set_label(r'',fontsize=fontsize)
		cb.ax.tick_params(labelsize=fontsize)

		ind_mplotlib = ind_mplotlib+1

	# Plotting data - Stokes U
		vmx = 0.5
		vmn = -vmx
		ScaleZ=colors.Normalize(vmn,vmx)
		axmainArtist_U = axmain[ind_mplotlib].imshow(dataBlock_U,cmap='Greys', aspect = 'auto', vmin=vmn,vmax=vmx, interpolation= 'none', extent =[time_limit[0], time_limit[1], fbot, ftop])

	# Setting colorbar
		ax=axmainArtist_U.axes
		fig=ax.figure
		divider = axes_grid1.make_axes_locatable(ax)
		cb = fig.colorbar(axmainArtist_U, extend='both',ax=axmain[ind_mplotlib])
		cb.set_label(r'',fontsize=fontsize)
		cb.ax.tick_params(labelsize=fontsize)

		ind_mplotlib = ind_mplotlib +1

# Plotting data - Stokes V
#	vmx = np.percentile(dataBlock_V, percentiles[-1])
	vmx = 0.4
	vmn = -vmx
	ScaleZ=colors.Normalize(vmn,vmx)
	axmainArtist_V = axmain[ind_mplotlib].imshow(dataBlock_V,cmap='bwr', aspect = 'auto', vmin=vmn,vmax=vmx, interpolation= 'none', extent =[time_limit[0], time_limit[1], fbot, ftop])
#	axmainArtist_V = axmain[3].pcolormesh([time_limit[0], time_limit[1]], [fbot, ftop], dataBlock_V, norm = ScaleZ, cmap='bwr', vmin=vmn,vmax=vmx)

# Setting colorbar
	ax=axmainArtist_V.axes
	fig=ax.figure
	divider = axes_grid1.make_axes_locatable(ax)
	cb = fig.colorbar(axmainArtist_V, extend='both',ax=axmain[ind_mplotlib])
	cb.set_label(r'Degree of Circular Polarization',fontsize=fontsize)
	cb.ax.tick_params(labelsize=fontsize)

	ind_mplotlib = ind_mplotlib +1

	if full_stokes == True:
	# Plotting data - Stokes L
		if flux_limits:
			vmn, vmx = flux_limits
		else:
			vmx, vmn = np.percentile(dataBlock_L, percentiles)
		vmn = 0
		vmx = 0.5
		ScaleZ=colors.Normalize(vmn,vmx)
		axmainArtist_L = axmain[ind_mplotlib].imshow(dataBlock_L,cmap='Greys', aspect = 'auto', vmin=vmn,vmax=vmx, interpolation= 'none', extent =[time_limit[0], time_limit[1], fbot, ftop])


	# Setting colorbar
		ax=axmainArtist_L.axes
		fig=ax.figure
		divider = axes_grid1.make_axes_locatable(ax)
		cb = fig.colorbar(axmainArtist_L, extend='both',ax=axmain[ind_mplotlib])
		cb.set_label(r'',fontsize=fontsize)
		cb.ax.tick_params(labelsize=fontsize)

# Plotting data - Stokes V/I
#	axmain[2].set_ylabel("Frequency (MHz)")
#	title_3 = axmain[2].set_title("")
#	dataBlock_VI = dataBlock_V / dataBlock_I
#	vmx = np.percentile(dataBlock_VI, percentiles[-1])
#	vmn = -vmx
#	ScaleZ=colors.Normalize(vmn,vmx)
#	axmainArtist_VI = axmain[2].imshow(dataBlock_VI,cmap='bwr', aspect = 'auto', vmin=vmn,vmax=vmx, interpolation =None, extent =[time_limit[0], time_limit[1], fbot, ftop])
#	title_3.set_text(f"Stokes V/I - {t_init.strftime('%d %b %Y')}")
#	title_3.set_size(fontsize+2)

# Setting colorbar
#	ax=axmainArtist_VI.axes
#	fig=ax.figure
#	divider = axes_grid1.make_axes_locatable(ax)
#	cb = fig.colorbar(axmainArtist_VI, extend='both',ax=axmain[2])
#	cb.set_label(r'Intensity',fontsize=fontsize)
#	cb.ax.tick_params(labelsize=fontsize)


# Telling matplotlib that the x-axis is filled with datetime data.
# This line converts it from a float into a nice datetime string
	axmain[-1].xaxis_date()


# Formatting the date
	if time_delta.total_seconds() <= 5:
		dateFmt = mdates.DateFormatter('%S.%f')
	elif time_delta.total_seconds() < 600:
		dateFmt=mdates.DateFormatter('%H:%M:%S')
	else:
		dateFmt=mdates.DateFormatter('%H:%M')

	axmain[1].xaxis.set_major_formatter(dateFmt)

	if full_stokes == True:
		if time_delta.total_seconds() <= 5:
			title_1.set_text(f"Stokes I - {t_init.strftime('%d %b %Y %H:%M')}")
			title_2.set_text(f"Stokes Q - {t_init.strftime('%d %b %Y %H:%M')}")
			title_3.set_text(f"Stokes U - {t_init.strftime('%d %b %Y %H:%M')}")
			title_4.set_text(f"Stokes V - {t_init.strftime('%d %b %Y %H:%M')}")
			title_5.set_text(f"Linear Polarization L - {t_init.strftime('%d %b %Y %H:%M')}")
		else:
			title_1.set_text(f"Stokes I - {t_init.strftime('%d %b %Y')}")
			title_2.set_text(f"Stokes Q - {t_init.strftime('%d %b %Y')}")
			title_3.set_text(f"Stokes U - {t_init.strftime('%d %b %Y')}")
			title_4.set_text(f"Stokes V - {t_init.strftime('%d %b %Y')}")
			title_5.set_text(f"Linear Polarization L - {t_init.strftime('%d %b %Y')}")
		title_3.set_size(fontsize+2)
		title_4.set_size(fontsize+2)
		title_5.set_size(fontsize+2)

	else:
		if time_delta.total_seconds() <= 5:
			title_1.set_text(f"Stokes I - {t_init.strftime('%d %b %Y %H:%M')}")
			title_2.set_text(f"Stokes V - {t_init.strftime('%d %b %Y %H:%M')}")
		else:
			title_1.set_text(f"Stokes I - {t_init.strftime('%d %b %Y')}")
			title_2.set_text(f"Stokes V - {t_init.strftime('%d %b %Y')}")

	title_1.set_size(fontsize+2)
	title_2.set_size(fontsize+2)


# Setting fontisze of ticks and labels
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	axmain[-1].xaxis.label.set_size(fontsize)
	axmain[0].yaxis.label.set_size(fontsize)
	axmain[1].yaxis.label.set_size(fontsize)

	axmain[0].tick_params(axis='y',labelsize=fontsize)
	axmain[1].tick_params(axis='y',labelsize=fontsize)

	if full_stokes == True:
		axmain[2].yaxis.label.set_size(fontsize)
		axmain[3].yaxis.label.set_size(fontsize)
		axmain[4].yaxis.label.set_size(fontsize)
		axmain[2].tick_params(axis='y',labelsize=fontsize)
		axmain[3].tick_params(axis='y',labelsize=fontsize)
		axmain[4].tick_params(axis='y',labelsize=fontsize)

#	axmain[0].yaxis.label.set_size(fontsize)
#	axmain[1].yaxis.label.set_size(fontsize)
##	axmain[2].yaxis.label.set_size(fontsize)
	plt.tight_layout()
	plt.savefig(filename)

def rollingAverage(data, step = 8):
	rollingSum = np.cumsum(data)
	return rollingSum[step:] - rollingSum[:-step]

def decimate(data, step = 64):
	rollingSum = np.cumsum(data)
	return rollingSum[step::step] - rollingSum[:-step:step]


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Plot data contained within a sigproc filterbank.")

	parser.add_argument('-i', dest = 'input', required = True, help = "Input File Location")

	parser.add_argument('--deci', dest = 'deci', default = 64, type = int, help = "Default decimation factor")

	parser.add_argument('--time_start', dest = 't_init', default = None, type = str, help = "Time start of the plot (YYYYMMDDhhmmss)")
	parser.add_argument('--time_end', dest = 't_end', default = None, type = str, help = "Time end of the plot (YYYYMMDDhhmmss)")
	parser.add_argument('--frequency_limits', dest = 'frequency_limits', nargs = 2,type=float, default = None, help = "Plotting frequency limits")
	parser.add_argument('--percentiles', dest = 'percentiles', nargs = 2, type = int, default = [99, 1], help = "Plotting percentile limits")
	parser.add_argument('--flux_limits', dest ='flux_limits', nargs = 2, type = float, default = None, help = "Plotting limits")

	parser.add_argument('--plot_raw', dest = 'plot_raw', default = False, action = 'store_true', help = "Plot the raw data")
	parser.add_argument('--plot_deci', dest = 'plot_deci', default = False, action = 'store_true', help = "Plot the base decimated data")
	parser.add_argument('--plot_deci_norm', dest = 'plot_deci_norm', default = False, action = 'store_true', help = "Plot the base decimated data, normalised")
	parser.add_argument('--plot_norm', dest= 'plot_norm', default = False, action = 'store_true', help = "Plot the normalised data")
	parser.add_argument('--reverse_freq', dest = 'rev', default = False, action = 'store_true', help = "Reverse the frequency axis")
	parser.add_argument('--subtract_background', dest='subtract_background', default = False, action = 'store_true', help = "Subtract a background to the data")
	parser.add_argument('--downsample_frequency', dest='downsample_frequency', type = int, default = 1, help = "Downsample the dataset in frequency")

	parser.add_argument('--full_stokes', dest = 'full_stokes', default = False, action = 'store_true', help = "Plot all Stokes Parameters and Linear Polarization L")
	parser.add_argument("--title", dest = 'title', default = 'plot', help = "Plot title prefix")
	parser.add_argument("--figsize", dest = 'figsize', nargs = 2, type = int, default=(26,14), help = "Figure size")
	parser.add_argument("--fontsize", dest = 'fontsize', type = int, default = 14, help = "Font size for the plot")
	parser.add_argument("-o", dest = 'prefix', default = 'plot', help = "Plot file prefix")
	parser.add_argument("--colormap", dest='cmap', type=str, default = None, help = "Color map for the plot")
	args = parser.parse_args()


	if args.plot_raw == args.plot_deci == args.plot_deci_norm == False:
		raise RuntimeError("Failed to provide any task to perform. Exiting.")

# Reading the fil file enter by the user
	filReader = spp.FilReader(args.input)

# Calculating the time boundary (to load only appropriate data, not all) if t_init and/or t_end have been given by the user
	time = Time(filReader.header.tstart, format = "mjd")
	time_filReader = Time(filReader.header.tstart, format = "mjd")

	if args.t_init == None:
		readTimestamp = 0
		t_init_user = None
		if args.t_end == None:
			samplesPerBlock = filReader.header.nsamples
			t_end_user = None
		else:
			t_end_user = datetime.datetime.strptime(args.t_end, "%Y%m%d%H%M%S")
			time_end_user = Time(t_end_user, format="datetime")
			time_end_user.format = "mjd"
			time_delta = time_end_user - time_filReader
			time_delta.format = 'sec'
			samplesPerBlock = int(time_delta.value/filReader.header.tsamp)*4

	if args.t_init != None:
		t_init_user = datetime.datetime.strptime(args.t_init, "%Y%m%d%H%M%S")
		time_init_user = Time(t_init_user, format="datetime")
		time_init_user.format = "mjd"
		print(f"Init. time from the fil file: {time_filReader.isot}")
		print(f"Init. time from the user: {time_init_user.isot}")
		time_delta = time_init_user - time_filReader
		time_delta.format = 'sec'
		readTimestamp = int(time_delta.value/filReader.header.tsamp)*4
		if args.t_end == None:
			samplesPerBlock = filReader.header.nsamples - startingSamples
			t_end_user = None
		else:
			t_end_user = datetime.datetime.strptime(args.t_end, "%Y%m%d%H%M%S")
			time_end_user = Time(t_end_user, format="datetime")
			time_end_user.format = "mjd"
			time_delta = time_end_user - time_init_user
			time_delta.format = 'sec'
			samplesPerBlock = int(time_delta.value/filReader.header.tsamp)*4
			print(f"End. time from the user: {time_end_user.isot}")


	prefixFolder = os.path.dirname(args.prefix)
	if not os.path.exists(prefixFolder):
		os.makedirs(prefixFolder)


	if args.plot_deci or args.plot_deci_norm:
		samplesPerBlock += samplesPerBlock % args.deci
	print(f"We will be reading {samplesPerBlock} samples per block from block {readTimestamp}.")

# Loading data
	dataBlock_all = filReader.readBlock(readTimestamp, samplesPerBlock)

# Storing data by (auto-)correlation component
	P_AA = np.zeros([filReader.header.nchans,int(samplesPerBlock/4)])
	P_BB = np.zeros([filReader.header.nchans,int(samplesPerBlock/4)])
	P_AB = np.zeros([filReader.header.nchans,int(samplesPerBlock/4)])
	P_BA = np.zeros([filReader.header.nchans,int(samplesPerBlock/4)])

	P_AA = dataBlock_all[:, 0::4]
	P_BB = dataBlock_all[:, 1::4]
	P_AB = dataBlock_all[:, 2::4]
	P_BA = dataBlock_all[:, 3::4]
# calculating Stokes parameters (I, Q, U, V) & linear polarization L
	dataBlock_I = spp.Filterbank.FilterbankBlock(P_AA+P_BB,dataBlock_all.header)
	dataBlock_Q = spp.Filterbank.FilterbankBlock(P_AA-P_BB,dataBlock_all.header)
	dataBlock_U = spp.Filterbank.FilterbankBlock(2*P_AB,dataBlock_all.header)
	dataBlock_V = spp.Filterbank.FilterbankBlock(-2*P_BA*10,dataBlock_all.header)
	dataBlock_L = spp.Filterbank.FilterbankBlock(np.sqrt((P_AA-P_BB)**2+(2*P_AB)**2),dataBlock_all.header)

# Downsampling the data and normalizing them if asked by the user
	if args.plot_deci or args.plot_deci_norm:
		dataBlock_I = dataBlock_I.downsample(tfactor = args.deci)
		dataBlock_V = dataBlock_V.downsample(tfactor = args.deci)
		bandpass = dataBlock_I.get_bandpass()
		if args.plot_deci_norm and args.subtract_background == False:
			dataBlock_I = dataBlock_I.normalise()

	if args.plot_norm and args.plot_deci_norm == False:
		dataBlock_I = dataBlock_I.normalise()
	else:
		bandpass = None

	if args.downsample_frequency != 1:
		dataBlock_I = dataBlock_I.downsample(ffactor = args.downsample_frequency)
		dataBlock_Q = dataBlock_Q.downsample(ffactor = args.downsample_frequency)
		dataBlock_U = dataBlock_U.downsample(ffactor = args.downsample_frequency)
		dataBlock_V = dataBlock_V.downsample(ffactor = args.downsample_frequency)
		dataBlock_L = dataBlock_L.downsample(ffactor = args.downsample_frequency)
	print(dataBlock_I.shape)

# Setting frequency limits if asked by the user
	if args.frequency_limits:
#		freq_array = np.array([dataBlock_all.header.ftop - j * dataBlock_all.header.bandwidth/(dataBlock_all.header.nchans-1) for j in range(dataBlock_all.header.nchans)])
		print(dataBlock_I.header)
		freq_array = np.array([dataBlock_I.header.ftop - j * np.abs(dataBlock_I.header.bandwidth*args.downsample_frequency)/(dataBlock_I.header.nchans-1) for j in range(dataBlock_I.header.nchans)])
		dataBlock_I = dataBlock_I[(np.where((freq_array > args.frequency_limits[0]) & (freq_array < args.frequency_limits[1])))[0],:]
		dataBlock_Q = dataBlock_Q[(np.where((freq_array > args.frequency_limits[0]) & (freq_array < args.frequency_limits[1])))[0],:]
		dataBlock_U = dataBlock_U[(np.where((freq_array > args.frequency_limits[0]) & (freq_array < args.frequency_limits[1])))[0],:]
		dataBlock_V = dataBlock_V[(np.where((freq_array > args.frequency_limits[0]) & (freq_array < args.frequency_limits[1])))[0],:]
		dataBlock_L = dataBlock_L[(np.where((freq_array > args.frequency_limits[0]) & (freq_array < args.frequency_limits[1])))[0],:]
		freq_array = (freq_array[np.where((freq_array > args.frequency_limits[0]) & (freq_array < args.frequency_limits[1]))])
		print(freq_array.shape)
		args.frequency_limits = (freq_array[-1],freq_array[0])
	else:
                args.frequency_limits = [dataBlock_all.header.fbottom,dataBlock_all.header.ftop]

# Subtracting a background if asked by the user
	if args.subtract_background:
		for ifreq in range(dataBlock_I.shape[0]):
			bck=10**(np.mean(np.log10(dataBlock_I[ifreq,:])))
			dataBlock_I[ifreq,:]=dataBlock_I[ifreq,:]/bck
			dataBlock_I[ifreq,:]=20*np.log10(dataBlock_I[ifreq,:])
			dataBlock_I[ifreq,:]=np.nan_to_num(dataBlock_I[ifreq,:], copy=True,nan=0,posinf=None,neginf=None)
#			bck_V=np.mean(dataBlock_V[ifreq,:])
#			dataBlock_V[ifreq,:]=dataBlock_V[ifreq,:]-bck_V


# Reversing frequency axis is asked by the user
	if args.rev:
		dataBlock_I = dataBlock_I[-1::-1, ...]
		dataBlock_V = dataBlock_V[-1::-1, ...]
		dataBlock_Q = dataBlock_Q[-1::-1, ...]
		dataBlock_U = dataBlock_U[-1::-1, ...]
		dataBlock_L = dataBlock_L[-1::-1, ...]

# Calling function to plot the data
	if args.t_init and args.t_end:
		print(f"t_init: {t_init_user} and t_end: {t_end_user}")
	if args.plot_raw:
		plot_data(dataBlock_I, dataBlock_Q, dataBlock_U, dataBlock_V, dataBlock_L, t_init_user, t_end_user, args.title, args.prefix, readTimestamp, args.percentiles, args.flux_limits, args.frequency_limits, args.figsize, args.fontsize, reverse = args.rev, cmap = args.cmap, full_stokes = args.full_stokes)
	if args.plot_norm and args.plot_deci == False:
		plot_data(dataBlock_I,dataBlock_Q, dataBlock_U, dataBlock_V, dataBlock_L, t_init_user, t_end_user, f"{args.title} (norm)", f"{args.prefix}_norm", readTimestamp, args.percentiles, args.flux_limits, args.frequency_limits, args.figsize, args.fontsize, cmap = args.cmap, full_stokes = args.full_stokes)
	if args.plot_deci or args.plot_deci_norm:
		plot_data(dataBlock_I,dataBlock_Q, dataBlock_U, dataBlock_V, dataBlock_L, t_init_user, t_end_user, f"{args.title} (Decimated x {args.deci})", f"{args.prefix}_deci_{args.deci}", readTimestamp, args.percentiles, args.flux_limits, args.frequency_limits, args.figsize, args.fontsize, cmap = args.cmap, full_stokes = args.full_stokes)
