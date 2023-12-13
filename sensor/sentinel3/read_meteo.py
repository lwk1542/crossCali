# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/13 15:35
@FileName: read_meteo.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import netCDF4 as nc
import os
import numpy as np
from scipy import interpolate


def read_nc(ds):
    value = ds[:, :] * 1.
    try:
        Fillvalue = ds.getncattr("_FillValue")
        value[value == Fillvalue] = np.nan
    except:
        pass
    try:
        add_offset = ds.getncattr("add_offset")
        value = value * add_offset
    except:
        pass
    try:
        scale_factor = ds.getncattr("scale_factor")
        value = value + scale_factor
    except:
        pass
    try:
        valid_max = ds.getncattr("valid_max")
        valid_min = ds.getncattr("valid_min")
        value[value < valid_min] = np.nan
        value[value > valid_max] = np.nan
    except:
        pass
    return value


def tie_correc(data_org, shape_target):
    shape = data_org.shape
    rows, columns = shape[0], shape[1]
    column_org = np.linspace(0, shape_target[1], num=columns, endpoint=True)
    rows_org = np.linspace(0, shape_target[0], num=rows, endpoint=True)
    rows_intp = np.arange(0, shape_target[0], 1)
    column_intp = np.arange(0, shape_target[1], 1)
    x, y = np.meshgrid(column_org, rows_org)
    x_, y_ = np.meshgrid(column_intp, rows_intp)
    if shape.__len__() == 3:
        data_new = np.zeros(shape=(shape_target[0], shape_target[1], shape[2]))+np.nan
        for i in range(shape[2]):
            data_new[:, :, i] = interpolate.griddata((y.flatten(), x.flatten()), data_org[:, :, i].flatten(), (y_, x_),
                                                     method='linear')
    else:
        data_new = interpolate.griddata((y.flatten(), x.flatten()), data_org.flatten(), (y_, x_), method='linear')
    return data_new


def read(in_file, out_shape):
    ds = nc.Dataset(os.path.join(in_file, "tie_meteo.nc"), mode="r")
    # temper = ds["atmospheric_temperature_profile"]
    humidity = ds["humidity"]
    pressure = ds["sea_level_pressure"]
    water_vapor = ds["total_columnar_water_vapour"]
    wind = ds["horizontal_wind"]
    oz = ds["total_ozone"]
    para = []
    for i in [humidity, pressure, water_vapor, oz]:
        tem_1 = read_nc(i)
        para.append(tie_correc(tem_1, out_shape))

    tem_1 = read_nc(wind)
    w12 = tie_correc(tem_1, out_shape)
    w1 = w12[:, :, 0]
    w2 = w12[:, :, 1]
    windspeed = np.sqrt(w1 ** 2 + w2 ** 2)  # 单位 m/s
    windspeed[windspeed < 0] = 0
    winddirection = np.arctan2(w1, w2) * 180 / np.pi
    para.append(windspeed)
    para.append(winddirection)

    return para
