# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7
@file: rayleigh_rad.py
@time: 2021/1/22 14:28
@desc: 非吸收性气体(瑞利散射)
"""

import numpy as np
from netCDF4 import Dataset
# from pyhdf.SD import SD, SDC
from scipy import interpolate
from l2gen import general


def rayleigh(rayleigh_lut_path=None, sza=None, vza=None, saa=None, vaa=None,reaa=None, windspeed=None,
             pressure=None, F0=None):
    """
    Args:
        rayleigh_lut_path (): 瑞利散射查找表路径
        sza (): 太阳天顶角
        vza (): 遥感器天顶角
        vaa ():
        saa ():
        windspeed (): 风速
        pressure (): 大气压
    Returns:
        Lr_i,Lr_q,Lr_q   8个波段的瑞利散射辐亮度的i q u分量，shape是（行数*像元数*波段数）
    """
    if reaa is None:
        reaa = vaa - 180 - saa
    reaa[reaa < -180] = reaa[reaa < -180] + 360
    reaa[reaa > 180] = reaa[reaa > 180] - 360

    # windspeed[windspeed > 25] = 25  # 将所有风速大于25m/s的数据截断到25m/s, 防止查找表边界溢出
    sza[sza > 88] = 88  # 将所有太阳高度角大于88°的数据截断到88°, 防止查找表边界溢出（定位数据有问题，或者晚上的数据混进来要保证执行）
    windspeed[np.isnan(windspeed)] = np.nanmean(windspeed)
    windspeed[windspeed < 0] = 0

    mu0 = np.cos(sza / 180 * np.pi)
    mu = np.cos(vza / 180 * np.pi)
    airmass = 1 / mu0 + 1 / mu

    rayleigh_lut = general.get_filelist(rayleigh_lut_path, 'rayleigh', 'iqu.hdf')
    print(rayleigh_lut)
    Taur = np.zeros(shape=(rayleigh_lut.__len__()))
    Lr_i = np.zeros(shape=(sza.shape[0], sza.shape[1], rayleigh_lut.__len__()))

    for i in range(rayleigh_lut.__len__()):
        rayDtset = Dataset(rayleigh_lut[i])
        taur = rayDtset.variables['taur'][:]
        # depol = rayDtset.variables['depol'][:]
        senz = rayDtset.variables['senz'][:]
        solz = rayDtset.variables['solz'][:]
        try:
            wind = rayDtset.variables['wind'][:]
            wind_inter_index = 0
        except:
            wind = rayDtset.variables['sigma'][:]
            wind_inter_index = 1
        i_ray = rayDtset.variables['i_ray'][:]
        # q_ray = rayDtset.variables['q_ray'][:]
        # u_ray = rayDtset.variables['u_ray'][:]

        Taur[i] = taur.data  # 每个通道的瑞利的光学厚度给出来,元组循环

        Norder0 = np.zeros_like(sza)
        Norder1 = np.ones_like(vza)
        Norder2 = Norder0 + 2

        # I  分量,注意插值的方法
        if wind_inter_index == 1:
            windspeed = 0.0731 * np.sqrt(windspeed)
        windspeed[windspeed > np.max(wind)] = np.max(wind)
        ray_i0 = interpolate.interpn(
            (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
            np.stack([windspeed, sza, Norder0, vza], axis=2), method='linear')
        ray_i1 = interpolate.interpn(
            (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
            np.stack([windspeed, sza, Norder1, vza], axis=2), method='linear')
        ray_i2 = interpolate.interpn(
            (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
            np.stack([windspeed, sza, Norder2, vza], axis=2), method='linear')
        #  L (l,h ,h,Dw)=L(0)(l,h ,h)+2 SIGMA L(m)(l,h ,h)cosmDw
        ray_i = ray_i0 + ray_i1 * np.cos(reaa / 180 * np.pi) + ray_i2 * np.cos(2 * reaa / 180 * np.pi)
        # # Q 分量
        # ray_q0 = interpolate.interpn(
        #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), q_ray,
        #     np.stack([windspeed, sza, Norder0, vza], axis=2))
        # ray_q1 = interpolate.interpn(
        #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), q_ray,
        #     np.stack([windspeed, sza, Norder1, vza], axis=2))
        # ray_q2 = interpolate.interpn(
        #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), q_ray,
        #     np.stack([windspeed, sza, Norder2, vza], axis=2))
        #
        # ray_q = ray_q0 + ray_q1 * np.cos(reaa / 180 * np.pi) + ray_q2 * np.cos(2 * reaa / 180 * np.pi)
        #
        # # U分量
        # ray_u0 = interpolate.interpn(
        #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), u_ray,
        #     np.stack([windspeed, sza, Norder0, vza], axis=2))
        # ray_u1 = interpolate.interpn(
        #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), u_ray,
        #     np.stack([windspeed, sza, Norder1, vza], axis=2))
        # ray_u2 = interpolate.interpn(
        #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), u_ray,
        #     np.stack([windspeed, sza, Norder2, vza], axis=2))
        #
        # ray_u = ray_u0 + ray_u1 * np.cos(reaa / 180 * np.pi) + ray_u2 * np.cos(2 * reaa / 180 * np.pi)

        #  from Wang menghua ;is from Seadas
        p0 = 1013.25  # 单位hpa   百帕
        x = (-(0.6543 - 1.608 * taur) + (0.8192 - 1.2541 * taur) * np.log(airmass)) * taur * airmass
        fac = ((1.0 - np.exp(-x * pressure / p0)) / (1.0 - np.exp(-x)))  # 气压校正参数
        Lr_i[:, :, i] = ray_i * fac * F0[i]

    return Lr_i