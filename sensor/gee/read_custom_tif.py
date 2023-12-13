# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: read_custom_tif.py
@time: 2022/1/11 14:51
@desc:
"""
import os
import numpy as np
import readtif
from ObservationGeometry import solar_azimuth, solar_zenith


def clipped_oli_info(infile):
    # 自己从gee上下载的lc08 oli数据，不具备通用性，根据自己的情况进行设置
    [lon_grid, lat_grid, data] = readtif.run(infile)

    bfile = os.path.basename(infile)
    year = int(bfile[-12:-8])
    yeararr = np.full_like(lon_grid, fill_value=year)
    month = int(bfile[-8:-6])
    montharr = np.full_like(lon_grid, fill_value=month)
    day = int(bfile[-6:-4])
    dayarr = np.full_like(lon_grid, fill_value=day)
    hourarr = np.full_like(lon_grid, fill_value=2.)
    minutearr = np.full_like(lon_grid, fill_value=15.)
    secondarr = np.full_like(lon_grid, fill_value=0.)
    sza = solar_zenith.get_zenith(lon_grid, lon_grid, yeararr, montharr, dayarr, hourarr, minutearr, secondarr)
    saa = solar_azimuth.get_azimuth(lat_grid, lon_grid, yeararr, montharr, dayarr, hourarr, minutearr,
                                    secondarr)
    vza, vaa = np.full_like(lon_grid, fill_value=0.), np.full_like(lon_grid, fill_value=0.5)

    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 0, 1, None, 2, 3, 4, 5,
    nwvis = 4
    red = num_670
    return [sza, vza, saa, vaa, lat_grid, lon_grid, data, year, month, day, num_443, num_490, num_520, num_555, num_670,
            nirs_num, nirl_num, nwvis, red]