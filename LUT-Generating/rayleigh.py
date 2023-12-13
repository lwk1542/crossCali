# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: rayleigh.py
@time: 2021/11/11 9:06
@desc: 生成瑞利查找表
参考资料：https://oceancolor.gsfc.nasa.gov/docs/rsr/rsr_tables/
"""
import h5py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate


def taur():
    """
    瑞利光学厚度
    下载光谱连续的瑞利光学厚度和传感器光谱响应函数
    Returns:
    """
    return


def band_com():
    # 读取目标传感器的光谱响应函数
    return


def interpolate_func(x, y, x_new=None):
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
    print("1")
    y_new = f(x_new)
    x_new = x_new
    return [x_new, y_new]


def band_average_quantity(spectrum=None, spectrum_response_function=None, solar_spectrum=None):

    """
        Args:
            attenuation_spectrum (): 臭氧/no2 衰减函数
            spectrum_response_function (): 光谱响应函数
        Returns:各波段臭氧衰减率
    """
    wave = spectrum.iloc[:, 0]
    solar_wave = solar_spectrum.iloc[:, 0]
    srf_wave = spectrum_response_function.iloc[:, 0]
    # 波段数：
    bands = spectrum_response_function.shape[1] - 1
    baq = []
    for i in range(bands):
        print(i)
        srf_band = spectrum_response_function.iloc[:, 1 + i]
        wave, spectrum_new = interpolate_func(wave, spectrum.iloc[:, 1], x_new=srf_wave)
        wave, solar_spectrum_new = interpolate_func(solar_wave, solar_spectrum.iloc[:, 1], x_new=srf_wave)
        baq_ = np.nansum(spectrum_new * srf_band*solar_spectrum_new) / np.nansum(srf_band*solar_spectrum_new)
        baq.append(baq_)
    return baq


def calcu():
    parameterFile = r"D:\researchProject_lwk\git_repository\oceanColorAtmosphericCorrection\LUT-Generating"

    # 1.辐照度
    thuillier_F0_file = parameterFile+os.sep+"SRF"+os.sep+"Thuillier_F0.txt"
    thuiller_F0 = pd.read_csv(thuillier_F0_file, index_col=None, header=None, sep="\s+")

    # 2.光谱响应函数
    spectrum_response_function_file = parameterFile + os.sep + "SRF" + os.sep + "MODIST.txt"
    srf = pd.read_table(spectrum_response_function_file, index_col=None, header=None, sep="\\s+", skiprows=8)

    # 3. 参数光谱
    spectrum_file = parameterFile + os.sep + "SRF" + os.sep + "taur.txt"
    spectrum = pd.read_table(spectrum_file, index_col=None, sep="\\s+", header=None, skiprows=15)

    baq = band_average_quantity(spectrum=spectrum, spectrum_response_function=srf, solar_spectrum=thuiller_F0)


def srf_diff():
    """
    目标传感器与光谱响应函数的差异
    Returns:
    """
    parameterFile=r"D:\researchProject_lwk\git_repository\oceanColorAtmosphericCorrection\LUT-Generating"
    hicosrf = parameterFile+os.sep+"SRF"+os.sep+"iss_hico_RSR.nc"
    targetsrf = parameterFile+os.sep+"SRF"+os.sep+"MODIST.txt"
    F0=parameterFile+os.sep+"SRF"+os.sep+"Thuillier_F0.txt"

    f_h=h5py.File(hicosrf, mode="r")
    rsr=f_h["RSR"][()]
    bands=f_h["bands"][()]
    wavelength=f_h["wavelength"][()]

    f_t=pd.read_csv(targetsrf, sep='\s+', header=None, skiprows=8)

    rsr1=f_t.iloc[:, 1:].values
    wavelength1 = f_t.iloc[:, 0].values
    draw_spectrum(rsr, wavelength,rsr1, wavelength1)


def draw_spectrum(rsr, wavelength, rsr1, wavelength1):

    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 7
             }

    fig, axes_arr = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    ax = axes_arr
    x = wavelength
    for i in range(rsr.shape[0]):
        y = rsr[i, :]
        f1, = ax.plot(x, y, linewidth=0.2, linestyle='-')
    x1=wavelength1
    # for j in range(rsr1.shape[1]-3):
    #     y1 = rsr1[:, j]
    #     f2, = ax.plot(x1, y1, linewidth=2, linestyle='-', )

    plt.show()

    ax.set_xlabel(u'Wavelength (nm)', fontdict=font1)
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.tick_params(axis='x', direction='in', length=3, width=1, colors='black', grid_color=None,
                   grid_alpha=0.5, labelrotation=0)

    ax.set_ylabel(u'$Rrs$ $(sr^{-1})}$', fontdict=font1)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.tick_params(axis='y', direction='in', length=3, width=1, colors='black', labelrotation=90)

    xlabels = np.array([412, 443, 490, 520, 565, 670])
    ymin, ymax = ax.get_ylim()
    ax.set_xlim([350,1300])
    ylabels = np.append(np.arange(0, ymax, 0.01)[0], np.arange(0, ymax, 0.01))
    ylabels = np.array([0, 0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.set_xticks([412, 443, 490, 520, 565, 670])
    ax.set_xticklabels(xlabels, fontdict=font1)
    ax.set_yticklabels(ylabels, fontdict=font1)
    ax.legend((f1,), [u'Class 1'], loc='upper right', fontsize=6)

    figname = r'D:\researchProject_lwk\git_repository\oceanColorAtmosphericCorrection\LUT-Generating/RSR_HICO_300dpi.png'
    plt.savefig(figname, dpi=300)
    plt.savefig(figname[0:-10] + '600dpi.png', dpi=600)
    plt.show()
    plt.close()


if __name__ == '__main__':
    srf_diff()
    # calcu()