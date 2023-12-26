# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7
@file: rayleigh_rad.py
@time: 2021/1/22 14:28
@desc: 非吸收性气体(瑞利散射)
修改记录2022-11-12，目标：利用Numba和GPU进行加速

"""

import numpy as np
from scipy import interpolate
import numba


# @numba.jit(nopython=True, parallel=True)
# def rayleigh(raylut_info=None, sza=None, vza=None, saa=None, vaa=None, reaa=None, windspeed=None, pressure=None,
#              F0=None):
#     """
#     Args:
#         reaa:
#         F0:
#         raylut_info: 从查找表中读取到信息
#         rayleigh_lut_path (): 瑞利散射查找表路径
#         sza (): 太阳天顶角
#         vza (): 遥感器天顶角
#         vaa ():
#         saa ():
#         windspeed (): 风速
#         pressure (): 大气压
#     Returns:
#         Lr_i,Lr_q,Lr_q   8个波段的瑞利散射辐亮度的i q u分量，shape是（行数*像元数*波段数）
#     """
#     if reaa is None:
#         reaa = vaa - 180 - saa
#     reaa[reaa < -180] = reaa[reaa < -180] + 360
#     reaa[reaa > 180] = reaa[reaa > 180] - 360
#
#     windspeed[windspeed > 25] = 25  # 将所有风速大于25m/s的数据截断到25m/s, 防止查找表边界溢出
#     sza[sza > 88] = 88  # 将所有太阳高度角大于88°的数据截断到88°, 防止查找表边界溢出（定位数据有问题，或者晚上的数据混进来要保证执行）
#     windspeed[np.isnan(windspeed)] = np.nanmean(windspeed)
#     windspeed[windspeed < 0] = 0
#
#     mu0 = np.cos(np.deg2rad(sza))
#     mu = np.cos(np.deg2rad(vza))
#     airmass = 1 / mu0 + 1 / mu
#
#     taur_ = raylut_info["taur"]
#     senz = raylut_info["senz"]
#     solz = raylut_info["solz"]
#     sigma = raylut_info["sigma"]
#     i_ray_ = raylut_info["i_ray"]
#     numbands = taur_.size
#     # sigma_ = 0.0731 * np.sqrt(windspeed)  # 风速转成sigma
#     sigma_ = np.sqrt(0.00534 * windspeed)
#     sigma_[sigma_ > np.max(sigma)] = np.max(sigma)  # 避免超限
#     Lr_i = np.zeros(shape=(sza.shape[0], sza.shape[1], taur_.size))
#     for i in range(numbands):
#         taur = taur_[i]
#         i_ray = i_ray_[i]
#         Norder0 = np.zeros_like(sza)
#         Norder1 = np.ones_like(vza)
#         Norder2 = Norder0 + 2
#         # I  分量,注意插值的方法
#         ray_i0 = interpolate.interpn(
#             (sigma.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
#             np.stack([sigma_, sza, Norder0, vza], axis=2), method='linear')
#         ray_i1 = interpolate.interpn(
#             (sigma.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
#             np.stack([sigma_, sza, Norder1, vza], axis=2), method='linear')
#         ray_i2 = interpolate.interpn(
#             (sigma.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
#             np.stack([sigma_, sza, Norder2, vza], axis=2), method='linear')
#         #  L (l,h ,h,Dw)=L(0)(l,h ,h)+2 SIGMA L(m)(l,h ,h)cosmDw
#         # ray_i = ray_i0 * np.cos(np.deg2rad(0*reaa)) + 2*ray_i1 * np.cos(np.deg2rad(1 * reaa)) + 2*ray_i2 * np.cos(
#         #     np.deg2rad(2 * reaa))
#         ray_i = ray_i0 + ray_i1 * np.cos(reaa / 180 * np.pi) + ray_i2 * np.cos(2 * reaa / 180 * np.pi)
#         # # Q 分量
#         # ray_q0 = interpolate.interpn(
#         #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), q_ray,
#         #     np.stack([windspeed, sza, Norder0, vza], axis=2))
#         # ray_q1 = interpolate.interpn(
#         #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), q_ray,
#         #     np.stack([windspeed, sza, Norder1, vza], axis=2))
#         # ray_q2 = interpolate.interpn(
#         #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), q_ray,
#         #     np.stack([windspeed, sza, Norder2, vza], axis=2))
#         #
#         # ray_q = ray_q0 + ray_q1 * np.cos(reaa / 180 * np.pi) + ray_q2 * np.cos(2 * reaa / 180 * np.pi)
#         #
#         # # U分量
#         # ray_u0 = interpolate.interpn(
#         #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), u_ray,
#         #     np.stack([windspeed, sza, Norder0, vza], axis=2))
#         # ray_u1 = interpolate.interpn(
#         #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), u_ray,
#         #     np.stack([windspeed, sza, Norder1, vza], axis=2))
#         # ray_u2 = interpolate.interpn(
#         #     (wind.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), u_ray,
#         #     np.stack([windspeed, sza, Norder2, vza], axis=2))
#         #
#         # ray_u = ray_u0 + ray_u1 * np.cos(reaa / 180 * np.pi) + ray_u2 * np.cos(2 * reaa / 180 * np.pi)
#
#         #  from Wang menghua ;is from Seadas
#         p0 = 1013.25  # 单位hpa   百帕
#         x = (-(0.6543 - 1.608 * taur) + (0.8192 - 1.2541 * taur) * np.log(airmass)) * taur * airmass
#         fac = ((1.0 - np.exp(-x * pressure / p0)) / (1.0 - np.exp(-x)))  # 气压校正参数
#         # Lr_i[:, :, i] = F0[i] * mu0 * ray_i * fac / np.pi
#         Lr_i[:, :, i] = ray_i * fac * F0[i]
#
#     return Lr_i

def rayleigh(bands:list,raylut_infos:dict, sza:np.ndarray, vza:np.ndarray, saa:np.ndarray, vaa:np.ndarray,
             windspeed:np.ndarray, pressure:np.ndarray, F0:np.ndarray):
    """
    Args:
        reaa:
        F0:
        raylut_info: 从查找表中读取到信息
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
    # reaa = vaa.copy() - saa.copy()
    # reaa[reaa > 360] = reaa[reaa > 360] - 360
    # reaa[reaa < 0] = reaa[reaa < 0] + 360
    # reaa = abs(reaa - 180)
    phi = vaa.copy() - saa.copy()
    phi = phi + 180.
    phi_idx = np.where(phi < 180)
    phi[phi_idx] = phi[phi_idx] + 360
    phi_idx = np.where(phi > 180)
    phi[phi_idx] = phi[phi_idx] - 360
    reaa=phi

    # windspeed[windspeed > 25] = 25  # 将所有风速大于25m/s的数据截断到25m/s, 防止查找表边界溢出
    # sza[sza > 88] = 88  # 将所有太阳高度角大于88°的数据截断到88°, 防止查找表边界溢出（定位数据有问题，或者晚上的数据混进来要保证执行）
    # windspeed[np.isnan(windspeed)] = np.nanmean(windspeed)
    # windspeed[windspeed < 0] = 0

    mu0 = np.cos(np.deg2rad(sza))
    mu = np.cos(np.deg2rad(vza))
    airmass = 1 / mu0 + 1 / mu
    Lr_i = np.zeros(shape=(sza.shape[0], sza.shape[1], bands.size))
    for i, band in enumerate(bands):
        print(band)
        raylut_info = raylut_infos[str(int(band))]
        taur = raylut_info["taur"]
        senz = raylut_info["senz"]
        solz = raylut_info["solz"]
        sigma = raylut_info["sigma"]
        i_ray = raylut_info["i_ray"]
        # sigma_ = 0.0731 * np.sqrt(windspeed)  # 风速转成sigma
        sigma_ = np.sqrt(0.00534 * windspeed)   # https://forum.earthdata.nasa.gov/viewtopic.php?t=2235
        sigma_[sigma_ > np.max(sigma)] = np.max(sigma)  # 避免超限

        # 归一化，避免插值误差
        sigma, sigma_ = normalization(sigma, sigma_)
        solz, sza = normalization(solz, sza)
        senz, vza = normalization(senz, vza)

        Norder0 = np.zeros_like(sza)
        # I  分量,注意插值的方法
        ray_i0 = interpolate.interpn(
            (sigma.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
            np.stack([sigma_, sza, Norder0, vza], axis=2), method='linear')
        ray_i1 = interpolate.interpn(
            (sigma.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
            np.stack([sigma_, sza, Norder0+1, vza], axis=2), method='linear')
        ray_i2 = interpolate.interpn(
            (sigma.reshape(-1), solz.reshape(-1), np.arange(3).reshape(-1), senz.reshape(-1)), i_ray,
            np.stack([sigma_, sza,Norder0+2, vza], axis=2), method='linear')
        #  L (l,h ,h,Dw)=L(0)(l,h ,h)+2 SIGMA L(m)(l,h ,h)cosmDw
        # ray_i = ray_i0 * np.cos(np.deg2rad(0*reaa)) + 2*ray_i1 * np.cos(np.deg2rad(1 * reaa)) + 2*ray_i2 * np.cos(
        #     np.deg2rad(2 * reaa))
        # ray_i = ray_i0 + ray_i1 * np.cos(reaa / 180 * np.pi) + ray_i2 * np.cos(2 * reaa / 180 * np.pi)
        p0 = 1013.25
        ray_i_temp = ray_i0 * np.cos(2 * np.pi * (reaa * 0) / 360) + ray_i1 * np.cos(
            2 * np.pi * (reaa * 1) / 360) + ray_i2 * np.cos(2 * np.pi * (reaa * 2) / 360)
        pp0 = pressure / p0
        pplt0 = np.where(pp0 < 0.)
        pp0[pplt0] = 1.
        tauray = pp0[:, :] * taur.data
        fac = (1. - np.exp(-tauray / np.cos(np.pi * vza / 180))) / (1. - np.exp((-taur / np.cos(np.pi * vza / 180))))
        Lr_i[:, :, i] = ray_i_temp * fac * F0[i]

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
        # p0 = 1013.25  # 单位hpa   百帕
        # x = (-(0.6543 - 1.608 * taur) + (0.8192 - 1.2541 * taur) * np.log(airmass)) * taur * airmass
        # fac = ((1.0 - np.exp(-x * pressure / p0)) / (1.0 - np.exp(-x)))  # 气压校正参数
        # # Lr_i[:, :, i] = F0[i] * mu0 * ray_i * fac / np.pi
        # Lr_i[:, :, i] = ray_i * fac * F0[i]

    return Lr_i


def normalization(data1, data2):
    _range = np.max(data1) - np.min(data1)
    data2[data2 > np.nanmax(data1)] = np.nanmax(data1)
    data2[data2 < np.nanmin(data1)] = np.nanmin(data1)
    return (data1 - np.min(data1)) / _range, (data2 - np.min(data1)) / _range


if __name__ == '__main__':
    rayleigh()
