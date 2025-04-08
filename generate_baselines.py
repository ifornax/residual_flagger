#!/usr/bin/env python3

import argparse
import numpy as np
import math
import astropy.units as u

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord

def generate_enu_baselines(antenna_positions_enu):
    """
    Given an array of antenna positions in ENU (East, North, Up),
    return an array of baseline vectors in ENU for all antenna pairs.
    Baseline = pos_j - pos_i for j > i.
    Shape: (Nbaselines, 3).
    """
    n_ant = len(antenna_positions_enu)
    baselines_enu = []
    for i in range(n_ant):
        for j in range(i+1, n_ant):
            baseline = antenna_positions_enu[j] - antenna_positions_enu[i]
            baselines_enu.append(baseline)
    return np.array(baselines_enu)

def enu_to_uvw(enu_vector, obs_time, location, phase_center):
    """
    Convert a single ENU baseline vector to UVW for the given time/location/phase center.
    We'll compute the hour angle from LST - RA, and apply a standard rotation matrix.
    """
    # Retrieve lat/lon from EarthLocation
    lat = location.lat
    lon = location.lon

    # Local sidereal time
    lst = obs_time.sidereal_time('apparent', longitude=lon)
    ha = (lst - phase_center.ra).to(u.rad).value  # in radians
    dec_rad = phase_center.dec.to(u.rad).value

    return enu_to_uvw_via_matrix(enu_vector, ha, dec_rad)

def enu_to_uvw_via_matrix(enu_vector, hour_angle, dec):
    """
    Apply the standard rotation from an ENU baseline vector [E,N,U] to [u,v,w],
    given the source hour angle and declination in radians.

    The commonly used matrix (Thompson, Moran, & Swenson, "Interferometry and Synthesis in Radio Astronomy")
    can be written as:

        [u]   = [ -sin(HA)              cos(HA)              0     ] [ E ]
        [v]   = [ -sin(DEC)*cos(HA)    -sin(DEC)*sin(HA)     cos(DEC) ] [ N ]
        [w]   = [  cos(DEC)*cos(HA)     cos(DEC)*sin(HA)     sin(DEC) ] [ U ]

    We'll implement that directly below.
    """
    E, N, U = enu_vector  # unpack
    sin_ha = math.sin(hour_angle)
    cos_ha = math.cos(hour_angle)
    sin_dec = math.sin(dec)
    cos_dec = math.cos(dec)

    # Construct the rotation matrix
    rot = np.array([
        [-sin_ha,                cos_ha,               0.0     ],
        [-sin_dec*cos_ha,       -sin_dec*sin_ha,       cos_dec ],
        [ cos_dec*cos_ha,        cos_dec*sin_ha,       sin_dec ]
    ])

    # Multiply
    enu_arr = np.array([E, N, U])
    uvw = rot @ enu_arr
    return uvw  # (u, v, w) in meters

def main():
    parser = argparse.ArgumentParser(
        description="Generate ENU baselines from antenna positions and convert them to UVW using observation info."
    )
    parser.add_argument(
        "fits_file",
        type=str,
        help="Path to the FITS file containing observation metadata (e.g. RA_TARG, DEC_TARG, DATE-OBS, OBS_LAT, OBS_LON)."
    )
    parser.add_argument(
        "antenna_file",
        type=str,
        help="Path to a text/CSV file containing antenna ENU positions, one row per antenna: E  N  U (in meters)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output filename (e.g. CSV) to save baseline UVW data."
    )
    parser.add_argument(
        "--ra-key",
        type=str,
        default="RA_TARG",
        help="FITS header keyword for target RA (degrees). Default RA_TARG."
    )
    parser.add_argument(
        "--dec-key",
        type=str,
        default="DEC_TARG",
        help="FITS header keyword for target Dec (degrees). Default DEC_TARG."
    )
    parser.add_argument(
        "--dateobs-key",
        type=str,
        default="DATE-OBS",
        help="FITS header keyword for observation start time (UTC). Default DATE-OBS."
    )
    parser.add_argument(
        "--lat-key",
        type=str,
        default="OBS_LAT",
        help="FITS header keyword for observatory latitude (deg). Default OBS_LAT."
    )
    parser.add_argument(
        "--lon-key",
        type=str,
        default="OBS_LON",
        help="FITS header keyword for observatory longitude (deg). Default OBS_LON."
    )
    parser.add_argument(
        "--alt-key",
        type=str,
        default="OBS_ALT",
        help="FITS header keyword for observatory altitude (meters). Default OBS_ALT."
    )
    args = parser.parse_args()

    # 1) Read observation metadata from FITS
    hdul = fits.open(args.fits_file)
    hdr = hdul[0].header

    # RA, Dec in degrees
    ra_deg = hdr[args.ra_key]
    dec_deg = hdr[args.dec_key]

    # Obs time (UTC in ISO format, e.g. 2025-01-01T12:00:00)
    utc_start = hdr[args.dateobs_key]

    # Site location
    lat_deg = hdr.get(args.lat_key)
    lon_deg = hdr.get(args.lon_key)
    alt_m   = hdr.get(args.alt_key, 0.0)

    hdul.close()

    # Create Astropy objects
    location = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=alt_m*u.m)
    obs_time = Time(utc_start, format="isot", scale="utc")
    phase_center = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")

    # 2) Load antenna positions from text/CSV (each row: E  N  U in meters)
    # Example:
    #  0   0    0
    #  30  10   0
    #  ...
    antenna_positions_enu = np.loadtxt(args.antenna_file)

    # 3) Generate ENU baselines
    baselines_enu = generate_enu_baselines(antenna_positions_enu)

    # 4) Convert each baseline to UVW
    uvw_list = []
    for baseline_enu in baselines_enu:
        uvw_vec = enu_to_uvw(baseline_enu, obs_time, location, phase_center)
        uvw_list.append(uvw_vec)
    uvw_array = np.array(uvw_list)  # shape (Nbaselines, 3)

    # 5) Save baseline + UVW data to a file
    # We'll write columns: Ex, Ey, Ez, U, V, W
    # If you prefer CSV with commas, you can adapt. We'll do whitespace:
    out_data = np.hstack([baselines_enu, uvw_array])  # shape (Nbaselines, 6)
    header_line = "Ex(m)  Ey(m)  Ez(m)   U(m)   V(m)   W(m)"

    np.savetxt(args.output_file, out_data, header=header_line)
    print(f"Saved baseline ENU & UVW to: {args.output_file}")

if __name__ == "__main__":
    main()
