This is a placeholder for the codes I use to flag RFI from the residual images.
An implementation of Keegan's ideas.

fft_image_and_produce_outliers_dask.py - 
This code will automatically run from anywhere, use the conda environment specified and will take a fits images (residual) and do a FFT.
The FFT image is then saved as well as it will write out a csv file that contains the U,v where there are potential outliers.

This file will be used with the uv_outliers_match.py to give the antennas with most outliers.




