# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/28 16:43
@FileName: assistant.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate


def interpolate_fun(x, y, x_new=None):
    """
    插值函数
    """
    if type(x) is np.ndarray:
        x = x
    else:
        # dataframe
        x = x.values
    if type(y) is np.ndarray:
        y = y
    else:
        y = y.values

    f = interpolate.interp1d(x, y, kind="linear", bounds_error=False)
    y_new = f(x_new)
    x_new = x_new
    return [x_new, y_new]


def calculate_band_average(reference_spectrum=None, spectrum_response_function=None):
    """
    插值
    积分
    """
    reference_wave = reference_spectrum.iloc[:, 0]
    reference_response = reference_spectrum.iloc[:, 1]

    srf_wave = spectrum_response_function.iloc[:, 0]
    # 波段数：
    bands = spectrum_response_function.shape[1] - 1
    band_values = []
    for i in range(bands):
        srf_band = spectrum_response_function.iloc[:, 1 + i]
        wave, solar_spectrum_new = interpolate_fun(reference_wave, reference_response, x_new=srf_wave)
        band_values_ = np.nansum(solar_spectrum_new * srf_band) / np.nansum(srf_band)
        band_values.append(band_values_)
    return band_values