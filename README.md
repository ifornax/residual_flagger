This is a placeholder for the codes I use to flag RFI from the residual images.
An implementation of Keegan's and Tim's ideas.

The idea is if one has a residual image and want to extract low level RFI

## You will need a Python 3.10.0 environment and install the following packages:
- subprocess
- astropy
- dask
- numpy
- parse

It wil use you environment name e.g. here it is called grg:
CONDA_ENV_NAME = "grg"  # Replace with your Conda environment name
  
**fft_image_and_produce_outliers_dask.py

This code will automatically run from anywhere, use the conda environment specified and will take a fits images (residual) and do a FFT.
The FFT image is then saved as well as it will write out a csv file that contains the U,v where there are potential outliers.

This file will be used with the uv_outliers_match.py to give the antennas with most outliers.

**generate_baseline_for_obs.py**

This code will take a fits file and generate the expected baseline using information from the header.
The output of this file can then be used as input to uv_outliers_match.py


