"""
计算卫星传感器大气层顶波段平均下行太阳辐照度
cited：http://www.gtzyyg.com/article/2012/1001-070X/1001-070X-24-3-97.html
胡顺石, 张立福, 张霞, 王倩, 韩冰, 张楠. .卫星传感器波段平均太阳辐照度计算及可靠性分析[J]. 国土资源遥感, 2012,24(3): 97-102
HU Shun-shi, ZHANG Li-fu, ZHANG Xia, WANG Qian, HAN Bing, ZHANG Nan. .Calculation and Reliability Analysis of Satellite
Sensors Band Solar Irradiance[J]. REMOTE SENSING FOR LAND & RESOURCES,2012,24(3): 97-102
"""

import numpy as np
import pandas as pd
import os


def interpolate(x, y, x_new=None):
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


def calculate_band_average_irradiance(solar_spectrum=None, spectrum_response_function=None):
    """
    插值
    积分
    """
    solar_wave = solar_spectrum.iloc[:, 0]
    srf_wave = spectrum_response_function.iloc[:, 0]
    # 波段数：
    bands = spectrum_response_function.shape[1] - 1
    E0 = []
    for i in range(bands):
        srf_band = spectrum_response_function.iloc[:, 1 + i]
        wave, solar_spectrum_new = interpolate(solar_wave, solar_spectrum.iloc[:, 1], x_new=srf_wave)
        E0_ = np.nansum(solar_spectrum_new * srf_band) / np.nansum(srf_band)
        E0.append(E0_)
    return E0


def k(attenuation_spectrum=None, spectrum_response_function=None):
    """
    Args:
        attenuation_spectrum (): 臭氧/no2 衰减函数
        spectrum_response_function (): 光谱响应函数
    Returns:各波段臭氧衰减率
    """
    solar_wave = attenuation_spectrum.iloc[:, 0]
    srf_wave = spectrum_response_function.iloc[:, 0]
    # 波段数：
    bands = spectrum_response_function.shape[1] - 1
    E0 = []
    for i in range(bands):
        srf_band = spectrum_response_function.iloc[:, 1 + i]
        wave, solar_spectrum_new = interpolate(solar_wave, attenuation_spectrum.iloc[:, 1], x_new=srf_wave)
        E0_ = np.nansum(solar_spectrum_new * srf_band) / np.nansum(srf_band)
        E0.append(E0_)
    return E0


def crossection(attenuation_spectrum=None, spectrum_response_function=None):
    """
        Args:
            attenuation_spectrum (): 臭氧/no2 衰减函数
            spectrum_response_function (): 光谱响应函数
        Returns:各波段臭氧衰减率
        """
    solar_wave = attenuation_spectrum.iloc[:, 0]
    srf_wave = spectrum_response_function.iloc[:, 0]
    # 波段数：
    bands = spectrum_response_function.shape[1] - 1
    E0 = []
    for i in range(bands):
        srf_band = spectrum_response_function.iloc[:, 1 + i]
        wave, solar_spectrum_new = interpolate(solar_wave, attenuation_spectrum.iloc[:, 1], x_new=srf_wave)
        E0_ = np.nansum(solar_spectrum_new * srf_band) / np.nansum(srf_band)
        E0.append(E0_)
    return E0


if __name__ == '__main__':
    # 1 光谱响应函数
    # 1.1 h1c
    # spectrum_response_function_file = r'C:\Users\lwk15\Documents\GitHub\HY_project\parameterFile\RSR\HY1C_COCTS_RSR.txt'
    # srf = pd.read_table(spectrum_response_function_file, index_col=None, header=0, sep="\s+")
    # # 1.2 T MODIS
    # spectrum_response_function_file = r'C:\git_repository\atmosphericCorrection\RSR\MODIS_Terra_RSR'
    # srf = pd.read_table(spectrum_response_function_file, index_col=None, header=None, sep="\\s+",skiprows=8)
    # 1.3 A MODIS
    spectrum_response_function_file = r'/liwenkai/atmosphericCorrection/RSR/MODIS_Aqua_RSR'
    srf = pd.read_table(spectrum_response_function_file, index_col=None, header=None, sep="\\s+", skiprows=8)


    # 1.辐照度
    # thuillier_F0_file = r'D:\atmosphericCorrectionFile\Thuillier_F0.txt'
    # thuiller_F0 = pd.read_csv(thuillier_F0_file, index_col=None, header=None, sep="\s+")
    # E0 = calculate_band_average_irradiance(solar_spectrum=thuiller_F0, spectrum_response_function=srf)
    # print(E0)

    # # 2. 臭氧衰减率
    # ozonefile=r'C:\Users\lwk15\Documents\GitHub\HY_project\advanced\userPublicPack/Ozoneattenuationcoefficients'
    # ozone = pd.read_csv(ozonefile, index_col=None, header=None, sep=" ", skiprows=19)
    # koz=k(attenuation_spectrum=ozone, spectrum_response_function=srf)
    # print(koz)
    #
    # # 3. NO2衰减率
    # no2file = r'C:\Users\lwk15\Documents\GitHub\HY_project\advanced\userPublicPack/NO2absorption'
    # no2 = pd.read_csv(no2file, index_col=None, header=None, sep=" ", skiprows=19)
    # kno2 = k(attenuation_spectrum=no2, spectrum_response_function=srf)
    # print(kno2)

    # 4. 消光截面 http://igaco-o3.fmi.fi/ACSO/cross_sections.html

    o3file = r'C:\git_repository\atmosphericCorrection\RSR/SCIA_O3_Temp_cross-section_V4.1.csv'
    o3 = pd.read_csv(o3file, index_col=None, header=None, sep='\\s+', skiprows=20)
    k = crossection(attenuation_spectrum=o3, spectrum_response_function=srf)

    # NO2file = r'NO2absorptionCrossSection'
    # NO2 = pd.read_table(NO2file, index_col=None, header=None, sep='\\s+', skiprows=20)
    # k = crossection(attenuation_spectrum=NO2, spectrum_response_function=srf)

    print(k)



