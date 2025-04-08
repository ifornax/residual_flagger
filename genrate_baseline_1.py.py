#!/usr/bin/env python3

import argparse
import math
import numpy as np
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
    Convert a single ENU baseline vector [E,N,U] to [u,v,w] for the given time/location/phase center.
    We'll compute hour angle = LST - RA, then apply a standard rotation matrix.
    """
    # Retrieve lat, lon from EarthLocation
    lat = location.lat
    lon = location.lon

    # Local sidereal time
    lst = obs_time.sidereal_time('apparent', longitude=lon)
    ha = (lst - phase_center.ra).to(u.rad).value  # hour angle in radians
    dec_rad = phase_center.dec.to(u.rad).value

    return enu_to_uvw_via_matrix(enu_vector, ha, dec_rad)

def enu_to_uvw_via_matrix(enu_vector, hour_angle, dec):
    """
    Standard rotation from an ENU baseline vector [E,N,U] to [u,v,w],
    given source hour angle and declination (both in radians).

    Matrix from Thompson, Moran, & Swenson (Interferometry and Synthesis in Radio Astronomy):
    
        [u] = [ -sin(HA)            cos(HA)             0      ] [ E ]
        [v] = [ -sin(DEC)*cos(HA)  -sin(DEC)*sin(HA)    cos(DEC)] [ N ]
        [w] = [  cos(DEC)*cos(HA)   cos(DEC)*sin(HA)    sin(DEC)] [ U ]
    """
    E, N, U = enu_vector
    sin_ha = math.sin(hour_angle)
    cos_ha = math.cos(hour_angle)
    sin_dec = math.sin(dec)
    cos_dec = math.cos(dec)

    rot = np.array([
        [-sin_ha,                cos_ha,               0.0     ],
        [-sin_dec*cos_ha,       -sin_dec*sin_ha,       cos_dec ],
        [ cos_dec*cos_ha,        cos_dec*sin_ha,       sin_dec ]
    ])

    enu_arr = np.array([E, N, U])
    uvw = rot @ enu_arr
    return uvw  # (u, v, w) in meters

def prompt_float(msg):
    """
    Repeatedly prompt the user until they provide a valid float.
    """
    while True:
        val_str = input(msg)
        try:
            return float(val_str)
        except ValueError:
            print("Invalid entry. Please enter a numeric value.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline UVW coordinates from antenna ENU positions and FITS observation metadata."
    )
    parser.add_argument(
        "fits_file",
        type=str,
        help="Path to the FITS file with observation metadata (RA/Dec/DATE-OBS/etc.)."
    )
    parser.add_argument(
        "antenna_file",
        type=str,
        help="Path to a text/CSV file containing antenna ENU positions, one row per antenna: E  N  U (in meters)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output file (e.g. CSV) to save baseline ENU and UVW data."
    )

    # Optional custom header keys
    parser.add_argument("--ra-key", type=str, default="RA_TARG", help="Header key for RA (deg). Default RA_TARG.")
    parser.add_argument("--dec-key", type=str, default="DEC_TARG", help="Header key for Dec (deg). Default DEC_TARG.")
    parser.add_argument("--dateobs-key", type=str, default="DATE-OBS", help="Header key for date/time. Default DATE-OBS.")
    parser.add_argument("--lat-key", type=str, default="OBS_LAT", help="Header key for site lat (deg). Default OBS_LAT.")
    parser.add_argument("--lon-key", type=str, default="OBS_LON", help="Header key for site lon (deg). Default OBS_LON.")
    parser.add_argument("--alt-key", type=str, default="OBS_ALT", help="Header key for site altitude (m). Default OBS_ALT.")

    args = parser.parse_args()

    # Read observation metadata from FITS
    hdul = fits.open(args.fits_file)
    hdr = hdul[0].header

    # ----------------
    # RA
    # ----------------
    if args.ra_key in hdr:
        ra_deg = hdr[args.ra_key]
    else:
        ra_deg = prompt_float(f"'{args.ra_key}' not in FITS header. Enter RA in degrees: ")

    # ----------------
    # Dec
    # ----------------
    if args.dec_key in hdr:
        dec_deg = hdr[args.dec_key]
    else:
        dec_deg = prompt_float(f"'{args.dec_key}' not in FITS header. Enter Dec in degrees: ")

    # ----------------
    # DATE-OBS
    # ----------------
    if args.dateobs_key in hdr:
        utc_start = hdr[args.dateobs_key]
    else:
        utc_start = input(f"'{args.dateobs_key}' not in FITS header. Enter DATE-OBS in ISO format (e.g. 2025-01-01T12:00:00): ")

    # ----------------
    # Observatory lat, lon, alt
    # ----------------
    if args.lat_key in hdr:
        lat_deg = hdr[args.lat_key]
    else:
        lat_deg = prompt_float(f"'{args.lat_key}' not in FITS header. Enter latitude in degrees: ")

    if args.lon_key in hdr:
        lon_deg = hdr[args.lon_key]
    else:
        lon_deg = prompt_float(f"'{args.lon_key}' not in FITS header. Enter longitude in degrees: ")

    if args.alt_key in hdr:
        alt_m = hdr[args.alt_key]
    else:
        alt_m = prompt_float(f"'{args.alt_key}' not in FITS header. Enter altitude in meters: ")

    hdul.close()

    # Create Astropy objects
    location = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=alt_m*u.m)
    obs_time = Time(utc_start, format="isot", scale="utc")
    phase_center = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")

    # Load antenna positions from file
    # Each row: E  N  U (meters)
    antenna_positions_enu = np.loadtxt(args.antenna_file)

    # Generate ENU baselines
    baselines_enu = generate_enu_baselines(antenna_positions_enu)

    # Convert each baseline to UVW
    uvw_list = []
    for baseline_enu in baselines_enu:
        uvw_vec = enu_to_uvw(baseline_enu, obs_time, location, phase_center)
        uvw_list.append(uvw_vec)
    uvw_array = np.array(uvw_list)  # shape (Nbaselines, 3)

    # Save baseline + UVW data
    # We'll write 6 columns: Ex Ey Ez U V W
    out_data = np.hstack([baselines_enu, uvw_array])
    header_line = "Ex(m)  Ey(m)  Ez(m)   U(m)   V(m)   W(m)"

    np.savetxt(args.output_file, out_data, header=header_line)
    print(f"Saved baseline ENU & UVW to: {args.output_file}")

if __name__ == "__main__":
    main()
