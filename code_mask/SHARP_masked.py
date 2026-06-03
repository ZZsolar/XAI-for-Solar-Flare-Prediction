"""
This code is for calculating SHARP keys manually.
input: HMI data from: http://jsoc.stanford.edu/ajax/lookdata.html
       data should be put in the 'data' folder.
       We may use the following segments:

       [example filename]                 --> [description]
        hmi.sharp_cea_*.Br.fits            --> radial component of the magnetic field vector
        hmi.sharp_cea_*.Bt.fits            --> theta-component of the magnetic field vector
        hmi.sharp_cea_*.Bp.fits            --> phi-component of the magnetic field vector
        hmi.sharp_cea_*.Br_err.fits        --> error in radial component of the magnetic field vector
        hmi.sharp_cea_*.Bt_err.fits        --> error in theta-component of the magnetic field vector
        hmi.sharp_cea_*.Bp_err.fits        --> error in phi-component of the magnetic field vector
        hmi.sharp_cea_*.conf_disambig.fits --> bits indicate confidence levels in disambiguation result
        hmi.sharp_cea_*.bitmap.fits        --> bits indicate result of automatic detection algorithm
        hmi.sharp_cea_*.magnetogram.fits   --> line-of-sight component of the magnetic field

output: .txt file as a table, including the array of all SHARP parameters needed in time series.
         16 SHARP parameters are included:
         TOTUSJH: Total unsigned current helicity  \ H_{total} = \Sigma abs(B_z \cdot J_z)

original auther: Monica Bobra in 1 May 2015

modified by: Ran Hao in 09 Aug 2021

latest modified by: Z. Zheng in 23 Sep 2025
"""
# import modules

import pandas as pd
import math
from astropy.io import fits
import numpy as np
from skimage import filters
import os
import csv
import re
import sys
from pprint import pprint
from datetime import datetime
from Calculate_sharpkeys_masked import *
import argparse

import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Select mask method.")
    parser.add_argument('--method', type=str, choices=['ori', 'pil', 'mfr'], default="mfr",
                        help="Method to select CSV file: 'original', 'pil', or 'model-focused region'.")
    parser.add_argument('--num_workers', type=int, default=8, help="Set the number of cores")
    parser.add_argument('--parallel', type=str2bool, nargs='?', const=False, default=True,
                    help='Enable parallel processing (True or False)')
    if 'ipykernel' in sys.modules:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    if args.method == "ori":
        csv_path = "/data_work/param_data/SHARP_params.csv"
        mask_path = None
    elif args.method == "pil":
        csv_path = "/data_work/param_data/PIL_params.csv"
        mask_path = f'/data_work/mask_data/pil/'
    elif args.method == "mfr":
        csv_path = f"/data_work/param_data/MFR_params.csv"
        mask_path = f'/data_work/mask_data/mfr/'
    else:
        raise TypeError(args.__dict__)

    return args, csv_path, mask_path

def get_file_time(file_name):
    pattern = r'(\d{8}_\d{6})_TAI'
    match = re.search(pattern, file_name)
    if match:
        timestamp_raw = match.group(1)
        dt_object = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S")
        formatted_timestamp = dt_object.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_timestamp
    else:
        return None

def get_mask(mask_array):
    """Binarize Grad-CAM attribution map via Otsu threshold."""
    def mask_threshold(map):
        threshold = filters.threshold_otsu(map)
        return map > threshold
    mask_array0 = np.abs(mask_array)
    mask_map = mask_threshold(mask_array0)
    
    return mask_map

def get_data(input):
    """function: get_data

    This function reads the appropriate data and metadata.
    """
    harpnum, file_name = input
    file_time = get_file_time(file_name)
    def read_fits(file):
        hdu = fits.open(file)
        try:
            data = hdu[1].data
            header = hdu[1].header
        except Exception as e:
            data = hdu[0].data
            header = hdu[0].header
        return data, header
    
    data_path = f'/data_work/data_sharp_cea/sharp_{harpnum:05d}/'
    bz, _ = read_fits(data_path + file_name + ".Br.fits")
    by, _ = read_fits(data_path + file_name + ".Bt.fits")
    bx, _ = read_fits(data_path + file_name + ".Bp.fits")
    bz_err, _ = read_fits(data_path + file_name + ".Br_err.fits")
    by_err, _ = read_fits(data_path + file_name + ".Bt_err.fits")
    bx_err, _ = read_fits(data_path + file_name + ".Bp_err.fits")
    los, _ = read_fits(data_path + file_name + ".magnetogram.fits")
    header = pd.read_csv(data_path + file_name + ".header.csv")
    bitmap, _ = read_fits(data_path + file_name + ".bitmap.fits")

    if args.method == "ori":
        conf_disambig, _ = read_fits(data_path + file_name + ".conf_disambig.fits")
        mask_map = (conf_disambig >= 70) & (bitmap >= 30)
    elif args.method == "pil":
        mask_map = np.load(mask_path + file_name + ".pil.npy")
    elif args.method == "mfr":
        mask_array = np.load(mask_path + file_name + ".attr.npy")
        mask_map = get_mask(mask_array)
    else:
        raise TypeError(f"method is {args.method}")
    
    rsun_ref = header['RSUN_REF'].tolist()[0]
    dsun_obs = header['DSUN_OBS'].tolist()[0]
    rsun_obs = header['RSUN_OBS'].tolist()[0]
    cdelt1 = header['CDELT1'].tolist()[0]

    cdelt1_arcsec = (math.atan((rsun_ref * cdelt1 * radsindeg) / (dsun_obs))) * (1 / radsindeg) * (3600.)
    nx = bz.shape[1]
    ny = bz.shape[0]

    # LOS error: 6.4 G homogeneous noise (Liu et al. 2012, Sol. Phys.)
    los_err = np.ndarray(shape=(ny, nx), dtype=float)
    los_err.fill(6.4)
    by_flipped = -1.0 * (np.array(by))  # sign flip for CEA convention

    return [bz, by_flipped, bx, bz_err, by_err, bx_err, bitmap, nx, ny, rsun_ref, rsun_obs,
            cdelt1_arcsec, los, los_err, mask_map, file_time, harpnum, file_name]

def main_run_masked(input):
    
    [bz, by, bx, bz_err, by_err, bx_err, bitmap, nx, ny, rsun_ref, rsun_obs,
            cdelt1_arcsec, los, los_err, mask_map, file_time, harpnum, file_name] = get_data(input)
    mean_vf, mean_vf_err, count_mask = compute_abs_flux_masked(bz, bz_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map)
    if count_mask == 0:
        SHARP_row = [harpnum, file_time] + np.full(17, np.nan).tolist()
        print(f"{mask_map.sum(), (mask_map*bz).sum()}, sharp: {harpnum}, date_time: {file_time}, do not have any useful pixel")

    else:
        horiz = compute_bh(bx, by, bz, bx_err, by_err, nx, ny)
        bh, bh_err = horiz[0], horiz[1]
        mean_gamma, mean_gamma_err = compute_gamma_masked(bz, bh, bz_err, bh_err, mask_map)
        total = compute_bt(bx, by, bz, bx_err, by_err, bz_err, nx, ny)
        bt, bt_err = total[0], total[1]
        mean_derivative_bt, mean_derivative_bt_err = computeBtderivative_masked(bt, bt_err, nx, ny, mask_map)
        mean_derivative_bh, mean_derivative_bh_err = computeBhderivative_masked(bh, bh_err, nx, ny, mask_map)
        mean_derivative_bz, mean_derivative_bz_err = computeBzderivative_masked(bz, bz_err, nx, ny, mask_map)
        current = computeJz_masked(bx, by, bx_err, by_err, nx, ny)
        jz, jz_err, derx, dery = current[0], current[1], current[2], current[3]
        mean_jz, mean_jz_err, us_i, us_i_err = computeJzmoments_masked(jz, jz_err, derx, dery, rsun_ref, rsun_obs, cdelt1_arcsec, munaught, mask_map)
        mean_alpha, mean_alpha_err = computeAlpha_masked(jz, jz_err, bz, bz_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map)
        mean_ih, mean_ih_err, total_us_ih, total_us_ih_err, total_abs_ih, total_abs_ih_err = computeHelicity_masked(jz, jz_err, bz, bz_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map)
        totaljz, totaljz_err = computeSumAbsPerPolarity_masked(jz, jz_err, bz, rsun_ref, rsun_obs, cdelt1_arcsec, munaught, mask_map)
        potential = greenpot(bz, nx, ny)
        bpx, bpy = potential[0], potential[1]
        meanpot, meanpot_err, totpot, totpot_err = computeFreeEnergy_masked(bx_err, by_err, bx, by, bpx, bpy, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map)
        meanshear_angle, meanshear_angle_err, area_w_shear_gt_45 = computeShearAngle_masked(bx_err, by_err, bz_err, bx, by, bz, bpx, bpy, mask_map)
        Rparam, Rparam_err = computeR_masked(los, los_err, cdelt1_arcsec, mask_map)
        mean_derivative_blos, mean_derivative_blos_err = computeLOSderivative_masked(los, los_err, nx, ny, mask_map, bitmap, args.method)
        mean_lf, mean_lf_err, count_mask = compute_abs_flux_los_masked(los, los_err, rsun_ref, rsun_obs, cdelt1_arcsec, mask_map, bitmap, args.method)
        SHARP_row = [harpnum, file_name, total_us_ih, totpot, us_i, total_abs_ih, totaljz, mean_vf, mean_lf, meanpot, Rparam, meanshear_angle, mean_gamma, mean_derivative_bt, mean_derivative_bz, mean_derivative_bh, mean_ih, mean_jz, mean_alpha, area_w_shear_gt_45]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(SHARP_name)
        writer.writerow(SHARP_row)

    print(f"sharp: {harpnum}, date_time: {file_time}, method: {args.method}, task done, time now: {datetime.now()}")
    return None

args, csv_path, mask_path = parse_args()

radsindeg = np.pi/180.
munaught  = 0.0000012566370614
SHARP_name = ['HARPNUM', 'file_name', 'TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', \
              'SAVNCPP', 'USFLUX', 'USFLUXL', 'MEANPOT', 'R_VALUE', 'MEANSHR', 'MEANGAM', \
              'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZH', 'MEANJZD', 'MEANALP', 'SHRGT45']

print(args.__dict__, flush=True)

from concurrent.futures import ProcessPoolExecutor

def main():
    label_path = "../data/label.csv"
    print(f"Reading {label_path} ...", flush=True)
    df = pd.read_csv(label_path)
    n_tasks = len(df)
    print(f"Total tasks: {n_tasks}, output: {csv_path}", flush=True)

    if os.path.isfile(csv_path):
        os.remove(csv_path)
        print(f"Removed existing {csv_path}", flush=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(SHARP_name)
    print(f"Created {csv_path} with header.", flush=True)

    if args.parallel:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for _, row in df.iterrows():
                harpnum = row['HARPNUM']
                file_name = row['file_name']
                futures.append(executor.submit(main_run_masked, [harpnum, file_name]))
            print(f"Submitted {len(futures)} tasks.", flush=True)
            for i, future in enumerate(futures):
                try:
                    future.result()
                    if (i + 1) % 100 == 0 or i == 0:
                        print(f"Completed {i + 1}/{len(futures)} tasks.", flush=True)
                except Exception as e:
                    print(f"Task {i + 1}/{len(futures)} failed: {e}", flush=True)
                    raise
        return None
    else:
        print("Not parallel", flush=True)
        for i, (_, row) in enumerate(df.iterrows()):
            harpnum = row['HARPNUM']
            file_name = row['file_name']
            try:
                main_run_masked([harpnum, file_name])
            except Exception as e:
                print(f"Task {i + 1}/{n_tasks} failed: {e}", flush=True)
                raise
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Completed {i + 1}/{n_tasks} tasks.", flush=True)

    print("all tasks down", flush=True)

if __name__ == "__main__":
    main()