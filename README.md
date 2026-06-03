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

Run the three commands in order. Each step produces an output that feeds into the next.

---

### Step 1 — Extract FFT outliers from the residual image

`fft-outliers` takes a FITS residual image, performs a 2D FFT, writes the UV-domain FITS image, and saves a CSV of candidate outlier (u,v) coordinates.

```bash
fft-outliers <input_image.fits> <fft_output.fits> <outliers.csv> [--nsigma 5.0] [--outlier-fits <marked.fits>]
```

| Argument | Description |
|---|---|
| `input_image.fits` | Input FITS residual image |
| `fft_output.fits` | Output FITS image of the 2D FFT |
| `outliers.csv` | Output CSV of outlier (u,v) coordinates and amplitudes |
| `--nsigma` | Sigma threshold for outlier detection (default: 5.0) |
| `--outlier-fits` | Optional: output FITS image with outliers marked |

---

### Step 2 — Compute expected baselines from the array geometry

`generate-baselines` takes the FITS image (for phase-centre RA/Dec and DATE-OBS) and a text file of antenna ENU positions, then computes all baseline UV coordinates and the antenna pair (i, j) for each.

```bash
generate-baselines <input_image.fits> <antenna_positions.enu.txt> <baselines.csv>
```

| Argument | Description |
|---|---|
| `input_image.fits` | Same FITS image used in Step 1 (provides RA, Dec, DATE-OBS) |
| `antenna_positions.enu.txt` | Text file of antenna East/North/Up positions (one row per antenna, space-separated) |
| `baselines.csv` | Output CSV of per-baseline UV coordinates and antenna pairs |

---

### Step 3 — Match FFT outliers to baselines

`uv-outliers-match` takes the outlier CSV from Step 1 and the baseline CSV from Step 2, matches each outlier to its nearest baseline, and reports which antennas are most likely responsible.

```bash
uv-outliers-match <outliers.csv> <baselines.csv> <matched_output.csv>
```

| Argument | Description |
|---|---|
| `outliers.csv` | Outlier CSV produced by Step 1 |
| `baselines.csv` | Baseline CSV produced by Step 2 |
| `matched_output.csv` | Output CSV with each outlier matched to its closest antenna pair |

---

### Full example

```bash
fft-outliers image-residual.fits image-fft.fits image-fft-outliers.csv --nsigma 5.0

generate-baselines image-residual.fits MeerKAT.enu.txt image-baselines.csv

uv-outliers-match image-fft-outliers.csv image-baselines.csv image-fft-outliers-matched.csv
```
