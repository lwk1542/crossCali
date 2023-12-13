# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/24 9:01
@FileName: 处理RSR.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import pandas as pd
import glob
import os
import numpy as np
import scipy.interpolate as interpolate

def fy3d_mersi_name_index():
    name = {"FY3D_MERSI_SRF_CH01_Pub.txt": 470,
            "FY3D_MERSI_SRF_CH02_Pub.txt": 550,
            "FY3D_MERSI_SRF_CH03_Pub.txt": 650,
            "FY3D_MERSI_SRF_CH04_Pub.txt": 865,
            "FY3D_MERSI_SRF_CH05_Pub.txt": 1380,
            "FY3D_MERSI_SRF_CH06_Pub.txt": 1640,
            "FY3D_MERSI_SRF_CH07_Pub.txt": 2130,
            "FY3D_MERSI_SRF_CH08_Pub.txt": 412,
            "FY3D_MERSI_SRF_CH09_Pub.txt": 443,
            "FY3D_MERSI_SRF_CH10_Pub.txt": 490,
            "FY3D_MERSI_SRF_CH11_Pub.txt": 555,
            "FY3D_MERSI_SRF_CH12_Pub.txt": 670,
            "FY3D_MERSI_SRF_CH13_Pub.txt": 709,
            "FY3D_MERSI_SRF_CH14_Pub.txt": 746,
            "FY3D_MERSI_SRF_CH15_Pub.txt": 865,
            "FY3D_MERSI_SRF_CH16_Pub.txt": 905,
            "FY3D_MERSI_SRF_CH17_Pub.txt": 936,
            "FY3D_MERSI_SRF_CH18_Pub.txt": 940,
            "FY3D_MERSI_SRF_CH19_Pub.txt": 1030
            }
    return name


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

    f = interpolate.interp1d(x, y, kind="linear", bounds_error=False, fill_value=0)
    y_new = f(x_new)
    x_new = x_new
    return [x_new, y_new]


def calculate_band_average(spectrum_response_function=None):
    """
    插值
    积分
    """
    # 波段数：
    bands = int(spectrum_response_function.shape[1]/2)
    band_values = []
    wavelngth = ["wavelength(nm)_470", "wavelength(nm)_550", "wavelength(nm)_650", "wavelength(nm)_865",
                 "wavelength(nm)_1380", "wavelength(nm)_1640", "wavelength(nm)_2130", "wavelength(nm)_412",
                 "wavelength(nm)_443", "wavelength(nm)_490", "wavelength(nm)_555", "wavelength(nm)_670",
                 "wavelength(nm)_709", "wavelength(nm)_746", "wavelength(nm)_865", "wavelength(nm)_905",
                 "wavelength(nm)_936", "wavelength(nm)_940", "wavelength(nm)_1030"]
    _ = spectrum_response_function[wavelngth]
    min_, max_ = np.floor(_.min().min()), np.ceil(_.max().max())
    new_spectrum = np.arange(min_, max_+1, 1)

    df = pd.DataFrame({"Wavelength (nm)": new_spectrum})
    for i in range(bands):
        _ = spectrum_response_function.iloc[:, 2*i: 2 * i + 1+1]
        _.dropna(axis=0, how='any')
        srf_wave = _.iloc[:, 0]
        srf_band = _.iloc[:, 1]
        wave, solar_spectrum_new = interpolate_fun(srf_wave, srf_band, x_new=new_spectrum)
        df_ = pd.DataFrame({srf_band.name: solar_spectrum_new})
        df = pd.concat([df, df_], axis=1)

    return df


def fengyun():
    "风云卫星的光谱响应函数一个波段对应一个文件，需将其全部插值到一个一个波长函数上"
    rsr_path = r"D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\RSR\FY3D_MERSI_SRF_Pub\pub"
    files = glob.glob(rsr_path+os.sep+"FY3D_MERSI_SRF_CH*_Pub.txt")
    names = fy3d_mersi_name_index()
    df = pd.DataFrame()
    for i, file in enumerate(files):
        if i>18:
            continue
        print(os.path.basename(file))
        df_ = pd.read_table(file, sep="   ", index_col=None, header=None)
        wave_name = names[os.path.basename(file)]
        df_.columns = ["wavelength(nm)_"+str(wave_name), str(wave_name)]
        df = pd.concat([df, df_], axis=1)

    rsr_sta = calculate_band_average(spectrum_response_function=df)
    rsr_sta.to_excel(r"D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\RSR\FY3D_MERSI_SRF_Pub"+os.sep+
                     "fy3.xlsx")


if __name__ == '__main__':
    fengyun()