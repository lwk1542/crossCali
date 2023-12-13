# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: read_landsat8oli.py
@time: 2022/1/9 21:37
@desc:
"""
import glob
import os

import numpy as np

from atmospheric_correction.oceancolor_acnirv2.sharepy.readfile import readtif


def l1_metadata(infile):
    hearfile = glob.glob(infile + os.sep + "LC08_L1TP_*_MTL.txt")
    if hearfile.__len__() == 0:
        print("can find the metadata file in folder: {0}".format(infile + os.sep))
    f = open(hearfile[0], mode="r")
    lines = f.readlines()
    date = [i.split("=", 1) for i in lines if "DATE_ACQUIRED" in i][0][1].strip()
    year, month, day = int(date[0:4]), int(date[5:7]), int(date[8:])
    hour, minute, second = 10, 15, 0
    saa_ = float([i.split("=", 1) for i in lines if "SUN_AZIMUTH" in i][0][1].strip())
    sza_ = 90 - float([i.split("=", 1) for i in lines if "SUN_ELEVATION" in i][0][1].strip())

    return [year, month, day, hour, minute, second, saa_, sza_]


def oli_info(infile):
    band_list = [band for band in os.listdir(infile) if band[-4:] == '.TIF']
    for i, band in enumerate(band_list):
        [lon_grid, lat_grid, band_value] = readtif(infile + os.sep + band)
        if i == 0:
            data = np.empty(shape=(band_value.shape[0], band_value.shape[1], band_list.__len__() - 1))
        data[:, :, i] = band_value
    [year, month, day, hour, minute, second, saa_, sza_] = l1_metadata(infile)
    sza = np.full_like(band_value, fill_value=sza_)
    saa = np.full_like(band_value, fill_value=saa_)
    vza, vaa = np.full_like(lon_grid, fill_value=0.), np.full_like(lon_grid, fill_value=0.5)
    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 0, 1, None, 2, 3, 4, 5,
    nwvis = 4
    red = num_670
    return [sza, vza, saa, vaa, lat_grid, lon_grid, data, year, month, day, num_443, num_490, num_520, num_555, num_670,
            nirs_num, nirl_num, nwvis, red]


# test
if __name__ == '__main__':
    oli_info(r"F:\cali_spatial_vari\OLI\LC08_L1TP_064045_20200126_20200210_01_T1")
    l1_metadata(r"F:\cali_spatial_vari\OLI\LC08_L1TP_064045_20200126_20200210_01_T1")
