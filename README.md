[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/423628067.svg)](https://zenodo.org/badge/latestdoi/423628067)


# ILOFAR
Code to read and process Jupiter radio emissions observations from LOFAR data

The *plot_raw_data.py* routine will calculate and plot, from a sigproc filterbank file, the Stokes I (intensity) and V (degree of circular polarization) parameters (and optionnaly the Stokes Q and U parameters and the degree of Linear Polarization).


**How to us it**

*Required entries:*
* -i: input data path/filename location. The data needs to be contained within a sigproc filterbank file
* -o: output dynamic spectrum path/filename location (without extension type; it will be png format).

*Optionnal entries:*

* --time_start *YYYYMMDDhhmmss*: Time start of the plot (*string*)
* --time_end *YYYYMMDDhhmmss*: Time end of the plot (*string*)

* --frequency_limits *f1 f2*: Plotting frequency limits (*two float value*)
* --reverse_freq: Reverse the frequency axis

* --percentiles *max min*: Plotting percentile limits for the intensity flux (*two float values*)
* --flux_limits *v1 v2*: Plotting fixed limits for the intensity flux (*two float values*)

* --full_stokes: Plot all Stokes Parameters and Linear Polarization L

* --plot_raw: Plot the raw data
* --plot_norm: Plot the normalised data (using sigpyproc normalise() function)
* --plot_deci: Plot raw temporal downsampled (base decimated) data
* --deci *deci_value*: downsampled decimation factor
* --plot_deci_norm: Plot normalised templral downsampled data

* --subtract_background: normalized the data by subtracting a background (on background per frequency calculated on the flight)

* --downsample_frequency *downsample_value*: Downsample the dataset in frequency (*int value*)

* --title *title*: Plot title prefix (*string*)
* --figsize: Figure size
* --fontsize *value*: Font size for the plot (*int value*)
* --colormap *cmap_name*: Color map for the plot (*string*)

* --help: entry that will give the user the above information

*example*:

python3 plot_raw_data.py -i /path/to/datafile/filename.fil-o /path/to/outputfile/output_filename --plot_raw --plot_deci --deci 4 --subtract_background --time_start 20210608051030 --time_end 20210608051040 --frequency_limits 8 40 --flux_limits -15 19 --colormap 'viridis' --figsize 15 15  --fontsize 22

In this example, *raw* data from *filename.fil* will be plot and saved into *output_filename.png* file. The data will be *downsample in time (by a factor 4)*. Only data between in the temporal *20210608051030*-*20210608051040* and spectral *8-40* ranges will be plot. The Stokes I data will be plot using the *viridis* colomarp, and the *Intensity flux limits* are fixed between *-15* and *19* dB.

**Requirements**
* argparse 1.1
* astropy 4.2
* datetime
* matplotlib 3.3.3
* mpl_toolkits
* numpy 1.19.4
* os 
* sigpyproc 0.2.0 
* tqdm 4.53.0
