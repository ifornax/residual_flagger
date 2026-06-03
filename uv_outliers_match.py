#!/usr/bin/env python3

def main():
    import argparse
    import pandas as pd
    import numpy as np
    from scipy.spatial import cKDTree

    # 1) Argument parsing
    parser = argparse.ArgumentParser(
        description="This code will take an FFT outlier CSV file and the baseline for that observation, then find outliers."
    )
    parser.add_argument("fft_outliers", type=str, help="Path to the input FITS image generated from fft_and_outliers_dask.py.")
    parser.add_argument("baseline_file", type=str, help="Path of CSV generated from generate_baseline_2.py.")
    parser.add_argument("outlier_csv", type=str, help="Where to write outlier info CSV.")

    args = parser.parse_args()

    # 2) Load CSVs
    print(f"Loading FFT outliers from: {args.fft_outliers}")
    print(f"Loading baselines from: {args.baseline_file}")
    df_outliers = pd.read_csv(args.fft_outliers)
    df_baselines = pd.read_csv(args.baseline_file)

    # 3) Calculate statistics
    max_uv = np.sqrt(df_outliers['u_lambda'].max()**2 + df_outliers['v_lambda'].max()**2) * 0.21  # Using 1.4 GHz
    max_vis = df_outliers['amplitude'].max()
    p01, p05, p50, p95, p99, p999 = np.percentile(df_outliers['amplitude'].values, [1, 5, 50, 95, 99, 99.9])

    print("Max Vis Report")
    print(f"    Max |v| = {max_uv:5.2f} m")
    print(f"    Max |vis| = {max_vis:5.2f}")
    print(f"        5% = {p05:5.2f}")
    print(f"        50% = {p50:5.2f}")
    print(f"        95% = {p95:5.2f}")
    print(f"        99% = {p99:5.2f}")
    print(f"        99.9% = {p999:5.2f}")

    # Filter outliers above 99th percentile
    df_outliers = df_outliers[df_outliers['amplitude'] >= p99]

    # 4) Build KD-tree on baseline (u,v)
    uv_base = df_baselines[["u_lambda", "v_lambda"]].values
    tree = cKDTree(uv_base)

    # 5) Find nearest baseline for each outlier
    uv_out = df_outliers[["u_lambda", "v_lambda"]].values
    distances, indices = tree.query(uv_out)

    # 6) Create merged DataFrame
    df_matched = df_outliers.copy()
    df_matched["closest_u_lambda"] = df_baselines["u_lambda"].values[indices]
    df_matched["closest_v_lambda"] = df_baselines["v_lambda"].values[indices]
    df_matched["antenna_i"] = df_baselines["antenna_i"].values[indices]
    df_matched["antenna_j"] = df_baselines["antenna_j"].values[indices]
    df_matched["dist_uv"] = distances

    # 7) Filter by tolerance
    tolerance = 50.0
    df_matched["is_close_enough"] = df_matched["dist_uv"] < tolerance

    # 8) Save to CSV
    df_matched.to_csv(args.outlier_csv, index=False)
    print(f"Merged results saved to: {args.outlier_csv}")

    # 9) Get common antennas
    common_ant = np.intersect1d(df_matched['antenna_i'].values.tolist(), df_matched['antenna_j'].values.tolist())
    print(f"Common antennas: {common_ant}")

if __name__ == "__main__":
    main()