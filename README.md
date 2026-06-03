This is a placeholder for the codes I use to flag RFI from the residual images.
An implementation of Keegan's and Tim's ideas.

The idea is if one has a residual image and want to extract low level RFI

## Installation

Requires Python 3.10+. Install with pip (preferably inside a virtual environment):

```bash
pip install .
```

This installs all required dependencies:
- astropy
- dask
- numpy
- pandas
- scipy
- matplotlib

## Usage

**fft-outliers**

Takes a FITS residual image, performs a 2D FFT, saves the UV-domain FITS image, and writes a CSV of potential outlier (u,v) coordinates.

```bash
fft-outliers <input_image.fits> <fft_output.fits> <outliers.csv> [--nsigma 5.0] [--outlier-fits <marked.fits>]
```

**generate-baselines**

Takes a FITS file and generates the expected baselines from the header. The output CSV is used as input to `uv-outliers-match`.

```bash
generate-baselines <input_image.fits> <baselines.csv>
```

**uv-outliers-match**

Takes the FFT outlier CSV and the baseline CSV, matches each outlier to the nearest baseline, and reports which antennas are most likely responsible.

```bash
uv-outliers-match <outliers.csv> <baselines.csv> <matched_output.csv>
```
