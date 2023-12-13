# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: rot.py
@time: 2021/1/22 9:09
@desc: 瑞利光学厚度
"""

from netCDF4 import Dataset
import numpy as np
import glob
import os


def tau_r(bands=np.array([412,443,490,20,565,670,750,865]),ray_lut_path=r'H:\workspace\OCPS_HY\MODIS_LUTs\modist\rayleigh'):
    """
    瑞利光学厚度
    瑞利光学厚度只和波长有关, 但需要做大气路径和气压的校正，这和观测几何气压有关
    """
    files = glob.glob(ray_lut_path + os.sep + 'rayleigh*_iqu.hdf')
    taur=np.empty(shape=(bands.__len__(),1))
    for i,file in enumerate(files):
        # rayDtset = Dataset(file)
        taur[i,0]=Dataset(file).variables['taur'][()][0]
    return taur