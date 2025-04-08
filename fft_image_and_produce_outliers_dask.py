#!/usr/bin/env python3

import os
import sys
import subprocess

# Configuration
CONDA_ENV_NAME = "grg"  # Replace with your Conda environment name
SCRIPT_NAME = os.path.abspath(__file__)  # Full path to this script

def is_conda_env_active(env_name):
    """Check if the specified Conda environment is active."""
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    return current_env == env_name

def activate_and_run():
    """Activate the Conda environment and re-run the script with original arguments."""
    # Preserve the original command-line arguments
    args = " ".join(sys.argv[1:])  # Exclude sys.argv[0] (script name)
    cmd = f"conda run -n {CONDA_ENV_NAME} python {SCRIPT_NAME} {args}"
    
    print(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
    process.wait()
    sys.exit(process.returncode)

def make_uv_header(hdr_in):
    """
    Given the original sky-image header, return a modified header suitable
    for the (u,v) domain, including adjusting CTYPE, CDELT, etc.
    We assume a simple tangent projection in the original, and a square image.
    """
    import math

    hdr_out = hdr_in.copy()

    nx = hdr_out["NAXIS1"]
    ny = hdr_out["NAXIS2"]
    # Original pixel size in degrees (assuming CUNIT1='deg', etc.)
    cdelt_deg = hdr_out["CDELT1"]  # might be negative if RA axis is reversed
    pix_scale_deg = abs(cdelt_deg)

    # Overwrite with UV-lambda information
    hdr_out["CTYPE1"] = "U"
    hdr_out["CTYPE2"] = "V"
    hdr_out["CUNIT1"] = "lambda"
    hdr_out["CUNIT2"] = "lambda"
    hdr_out["CRVAL1"] = 0.0
    hdr_out["CRVAL2"] = 0.0

    # Compute pixel in uv-lambda:
    delta_uv = 1.0 / (pix_scale_deg * math.pi / 180.0 * nx)

    hdr_out["CDELT1"] = delta_uv
    hdr_out["CDELT2"] = delta_uv

    # Reference pixel in the center
    hdr_out["CRPIX1"] = (nx + 1) / 2.0
    hdr_out["CRPIX2"] = (ny + 1) / 2.0

    return hdr_out

def fft_image_with_dask(input_fits, output_fits):
    """
    1) Load input FITS image (2D, 3D, or 4D) with memmap=True.
    2) If needed, slice out the first plane to get a 2D array.
    3) Convert to a Dask array and rechunk so that FFT can proceed.
    4) Perform 2D FFT, shift DC to center, take absolute value.
    5) Write the result to output_fits with a modified UV header.
    Returns (vis_amp, uv_hdr).
    """
    from astropy.io import fits
    import dask.array as da
    import dask.array.fft as dafft


    with fits.open(input_fits, memmap=True) as hdul:
        hdr_in = hdul[0].header
        data_in = hdul[0].data

        # Slice to 2D if necessary
        if data_in.ndim == 4:
            # e.g. shape (stokes, freq, dec, ra)
            data_2d = data_in[0, 0, :, :]
        elif data_in.ndim == 3:
            # e.g. shape (stokes/freq, dec, ra)
            data_2d = data_in[0, :, :]
        else:
            # Already 2D
            data_2d = data_in

        # Convert to Dask array
        image_da = da.from_array(data_2d)

        # IMPORTANT: Rechunk so each FFT axis has only one chunk
        ny, nx = data_2d.shape
        image_da = image_da.rechunk((ny, nx))

        # Perform 2D FFT
        vis_da = dafft.fft2(image_da)
        vis_shifted_da = dafft.fftshift(vis_da)
        vis_amp_da = da.absolute(vis_shifted_da)

        # Bring into memory
        vis_amp = vis_amp_da.compute()

        # Construct UV header
        uv_hdr = make_uv_header(hdr_in)
        uv_hdr["NAXIS1"], uv_hdr["NAXIS2"] = vis_amp.shape

        # Write to new FITS
        hdu_out = fits.PrimaryHDU(vis_amp, header=uv_hdr)
        hdu_out.writeto(output_fits, overwrite=True)

    return vis_amp, uv_hdr

def find_outliers_dask(vis_amp, uv_hdr, output_csv, output_fits=None, nsigma=5.0):
    """
    Given a 2D amplitude array (vis_amp) in (u,v) space and its header,
    find pixels above mean + nsigma*std. Write them to CSV, optionally mark them in FITS.
    """
    import numpy as np 

    mean_val = np.mean(vis_amp)
    std_val = np.std(vis_amp)
    threshold = mean_val + nsigma * std_val

    outlier_mask = vis_amp > threshold
    outlier_indices = np.where(outlier_mask)
    out_count = len(outlier_indices[0])
    print(f"Found {out_count} outliers above {nsigma}σ.")

    ny, nx = vis_amp.shape

    crval1 = uv_hdr.get("CRVAL1", 0.0)
    crval2 = uv_hdr.get("CRVAL2", 0.0)
    cdelt1 = uv_hdr.get("CDELT1", 1.0)
    cdelt2 = uv_hdr.get("CDELT2", 1.0)
    crpix1 = uv_hdr.get("CRPIX1", (nx + 1) / 2)
    crpix2 = uv_hdr.get("CRPIX2", (ny + 1) / 2)

    yvals, xvals = outlier_indices

    # Convert (x,y) to (u,v)
    u_vals = crval1 + (xvals + 1 - crpix1) * cdelt1
    v_vals = crval2 + (yvals + 1 - crpix2) * cdelt2
    amp_vals = vis_amp[yvals, xvals]

    # Write CSV
    with open(output_csv, "w") as f:
        f.write("u_lambda,v_lambda,amplitude\n")
        for (u, v, amp) in zip(u_vals, v_vals, amp_vals):
            f.write(f"{u},{v},{amp}\n")
    print(f"Outlier list saved to {output_csv}")

    # Optionally mark them
    if output_fits is not None:
        marked_data = vis_amp.copy()
        replacement_val = marked_data.max() + std_val
        marked_data[outlier_mask] = replacement_val

        hdu_out = fits.PrimaryHDU(marked_data, header=uv_hdr)
        hdu_out.writeto(output_fits, overwrite=True)
        print(f"Marked outliers FITS saved to {output_fits}")

def main():

    import argparse
    import numpy as np


    parser = argparse.ArgumentParser(
        description="FFT a sky image (2D/3D/4D) in UV space via Dask, then find outliers."
    )
    parser.add_argument("input_image", type=str, help="Path to the input FITS image.")
    parser.add_argument("fft_image", type=str, help="Where to write the FFTed (U,V) FITS.")
    parser.add_argument("outlier_csv", type=str, help="Where to write outlier info CSV.")
    parser.add_argument(
        "--outlier-fits",
        type=str,
        default=None,
        help="Optional: path to write a FITS file marking outlier pixels.",
    )
    parser.add_argument(
        "--nsigma",
        type=float,
        default=5.0,
        help="Threshold = mean + nsigma*std (default 5).",
    )
    args = parser.parse_args()

    # 1) Do the 2D FFT in (u,v) (with a rechunk)
    vis_amp, uv_hdr = fft_image_with_dask(args.input_image, args.fft_image)
    print(f"Wrote FFTed image: {args.fft_image}")

    # 2) Find outliers and save results
    find_outliers_dask(
        vis_amp,
        uv_hdr,
        output_csv=args.outlier_csv,
        output_fits=args.outlier_fits,
        nsigma=args.nsigma
    )

if __name__ == "__main__":
    # Check if we're already in the target Conda environment
    if not is_conda_env_active(CONDA_ENV_NAME):
        print(f"Activating Conda environment '{CONDA_ENV_NAME}' and re-running script...")
        activate_and_run()
    else:
        print(f"Already in Conda environment: {CONDA_ENV_NAME}")
        main()