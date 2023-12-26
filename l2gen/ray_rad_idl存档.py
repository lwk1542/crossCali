# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/22 10:11
@FileName: ray_rad.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from pyhdf.SD import SD, SDC
import numpy as np
import os
import glob

def ray_rad(rayleigh_lut_path, press, sza, vza, phi, wavelength, ns, nl, doy,F0, lambdas, windspeed):
    """
    没有风速
    :param rayleigh_lut_path:
    :param press:
    :param sza:
    :param vza:
    :param phi:
    :param wavelength:
    :param ns:
    :param nl:
    :param doy:
    :param F0:
    :param lambdas:
    :return:
    """
    # lut = r'C:\Users\lwk15\Documents\Tencent Files\2939027854\FileRecv/rayleigh_modisa_443_iqu.hdf'

    phi = phi+180.
    phi_idx = np.where(phi < 180)
    phi[phi_idx]=phi[phi_idx]+360
    phi_idx = np.where(phi > 180)
    phi[phi_idx] = phi[phi_idx] - 360

    press0 = 1013.25

    lut_file = glob.glob(rayleigh_lut_path + os.sep + 'rayleigh_*' + str(int(wavelength)) + '_iqu.hdf')[0]

    f = SD(lut_file, SDC.READ)
    sun_j = (f.select('solz')).get()
    i_ray = (f.select('i_ray')).get()  # 瑞利散射的I成分
    tau_i = (f.select('taur')).get()
    ray_ang = (f.select('senz')).get()
    msun = nsun_ray = 45               # 45个太阳天顶角
    nrad_ray = 41                      # 41个卫星天顶角
    norder_ray = 3
    ray_for_i = np.empty(shape =[norder_ray, nrad_ray, nsun_ray])
    for j in range(msun):
        ray_for_i[0, :, j] = i_ray[0, j, 0, :]
        ray_for_i[1, :, j] = i_ray[0, j, 1, :]
        ray_for_i[2, :, j] = i_ray[0, j, 2, :]

    isun_low = np.int16(np.trunc(sza/2))
    isun_high = isun_low+1

    j1 = np.zeros(shape=[nl, ns], dtype=np.int16)
    j2 = np.zeros(shape=[nl, ns], dtype=np.int16)
    for k in range(nl):
        for n in range(ns):
            j = np.where(ray_ang >= vza[k, n])
            j2[k, n] = j[0][0]
            j1[k, n] = j2[k, n]-1

    raj1 = ray_ang[j1]
    raj2 = ray_ang[j2]

    rsh = sun_j[isun_high]
    rsl = sun_j[isun_low]
    hsun = rsh -rsl
    hview = raj2 - raj1
    p = (sza - rsl)/hsun
    q = (vza - raj1)/hview

    ray_i = np.zeros(shape=[nl, ns])
    for m in range(3):
        marr = np.zeros(shape=[nl, ns], dtype=np.int16)+m
        f00 = ray_for_i[marr, j1, isun_low]
        f10 = ray_for_i[marr, j1, isun_high]
        f01 = ray_for_i[marr, j2, isun_low]
        f11 = ray_for_i[marr, j2, isun_high]
        value = (1.-p)*(1.-q)*f00+p*q*f11+p*(1.-q)*f10+q*(1.-p)*f01
        ray_i = (value*np.cos(2*np.pi*(phi*m)/360))+ray_i

    # pp0 = np.nan(shape =[nl,ns])
    # fac = np.nan(shape =[nl,ns])
    tauray = np.empty(shape=[nl, ns])
    pp0 = press/press0
    pplt0 = np.where(pp0 < 0.)
    pp0[pplt0] = 1.
    tauray[:, :] = pp0[:, :]*tau_i
    fac = (1.-np.exp(-tauray/np.cos(np.pi*vza/180)))/(1.-np.exp((-tau_i/np.cos(np.pi*vza/180))))
    ray_i = ray_i*fac

    i = np.where(lambdas == wavelength)
    # FoBAR = np.nan(shape =[nl,ns])
    A = 1.00014
    B = 0.01671
    C = 0.9856002831
    D = 3.452868
    E = 360.
    F1 = 1./((A-B*np.cos(2.*np.pi*(C*doy-D)/E)-0.000014*np.cos(4.*np.pi*(C*doy-D)/E))**2)
    FoBAR  =F0[i]*F1
    ray_i = ray_i*FoBAR
    f.end()
    return ray_i


def ray_rad(rayleigh_lut_path, press, sza, vza, phi, wavelength, ns, nl, doy,F0, lambdas, windspeed):
    """
    没有风速
    :param rayleigh_lut_path:
    :param press:
    :param sza:
    :param vza:
    :param phi:
    :param wavelength:
    :param ns:
    :param nl:
    :param doy:
    :param F0:
    :param lambdas:
    :return:
    """
    # lut = r'C:\Users\lwk15\Documents\Tencent Files\2939027854\FileRecv/rayleigh_modisa_443_iqu.hdf'

    phi = phi+180.
    phi_idx = np.where(phi < 180)
    phi[phi_idx]=phi[phi_idx]+360
    phi_idx = np.where(phi > 180)
    phi[phi_idx] = phi[phi_idx] - 360

    press0 = 1013.25
    # lambdas = [412, 443, 490, 520, 565, 670, 750, 865]
    # lambdas = np.array([412, 442, 488, 530, 554, 645, 747, 857, 1242, 1640, 2130])
    # F0 = np.array([172.632, 187.622, 194.933, 185.747, 183.869, 157.811, 128.065, 97.174, 45.467])
    # koz = np.array([1.98, 3.19, 20.32, 68.38, 95.53, 73.82, 12.35, 2.35, 0.0])

    # lut_dir = ''
    # INF=[]
    # INF[0] = 'rayleigh_modisa_412_iqu.hdf'
    # INF[1] = 'rayleigh_modisa_443_iqu.hdf'
    # INF[2] = 'rayleigh_modisa_488_iqu.hdf'
    # INF[3] = 'rayleigh_modisa_531_iqu.hdf'
    # INF[4] = 'rayleigh_modisa_555_iqu.hdf'
    # INF[5] = 'rayleigh_modisa_645_iqu.hdf'
    # INF[6] = 'rayleigh_modisa_748_iqu.hdf'
    # INF[7] = 'rayleigh_modisa_859_iqu.hdf'
    # INF[8] = 'rayleigh_modisa_1240_iqu.hdf'
    # INF[9] = 'rayleigh_modisa_1640_iqu.hdf'
    # INF[10] = 'rayleigh_modisa_2130_iqu.hdf'
    # i = np.where(lambdas == wavelength)
    # lut_file = lut_dir + INF[i]
    # rayleigh_lut = general.get_filelist(rayleigh_lut_path, 'rayleigh', 'iqu.hdf')
    ind = np.where(lambdas == wavelength)[0][0]
    lut_file = glob.glob(rayleigh_lut_path + os.sep + 'rayleigh_*' + str(int(wavelength)) + '_iqu.hdf')[0]

    f = SD(lut_file, SDC.READ)
    sun_j = (f.select('solz')).get()
    i_ray = (f.select('i_ray')).get()  # 瑞利散射的I成分
    tau_i = (f.select('taur')).get()
    ray_ang = (f.select('senz')).get()
    msun = nsun_ray = 45               # 45个太阳天顶角
    nrad_ray = 41                      # 41个卫星天顶角
    norder_ray = 3
    ray_for_i = np.empty(shape =[norder_ray, nrad_ray, nsun_ray])
    for j in range(msun):
        ray_for_i[0, :, j] = i_ray[0, j, 0, :]
        ray_for_i[1, :, j] = i_ray[0, j, 1, :]
        ray_for_i[2, :, j] = i_ray[0, j, 2, :]

    isun_low = np.int16(np.trunc(sza/2))
    isun_high = isun_low+1

    j1 = np.zeros(shape=[nl, ns], dtype=np.int16)
    j2 = np.zeros(shape=[nl, ns], dtype=np.int16)
    for k in range(nl):
        for n in range(ns):
            j = np.where(ray_ang >= vza[k, n])
            j2[k, n] = j[0][0]
            j1[k, n] = j2[k, n]-1

    raj1 = ray_ang[j1]
    raj2 = ray_ang[j2]

    rsh = sun_j[isun_high]
    rsl = sun_j[isun_low]
    hsun = rsh -rsl
    hview = raj2 - raj1
    p = (sza - rsl)/hsun
    q = (vza - raj1)/hview

    ray_i = np.zeros(shape=[nl, ns])
    for m in range(3):
        marr = np.zeros(shape=[nl, ns], dtype=np.int16)+m
        f00 = ray_for_i[marr, j1, isun_low]
        f10 = ray_for_i[marr, j1, isun_high]
        f01 = ray_for_i[marr, j2, isun_low]
        f11 = ray_for_i[marr, j2, isun_high]
        value = (1.-p)*(1.-q)*f00+p*q*f11+p*(1.-q)*f10+q*(1.-p)*f01
        ray_i = (value*np.cos(2*np.pi*(phi*m)/360))+ray_i

    # pp0 = np.nan(shape =[nl,ns])
    # fac = np.nan(shape =[nl,ns])
    tauray = np.empty(shape=[nl, ns])
    pp0 = press/press0
    pplt0 = np.where(pp0 < 0.)
    pp0[pplt0] = 1.
    tauray[:, :] = pp0[:, :]*tau_i
    fac = (1.-np.exp(-tauray/np.cos(np.pi*vza/180)))/(1.-np.exp((-tau_i/np.cos(np.pi*vza/180))))
    ray_i = ray_i*fac

    i = np.where(lambdas == wavelength)
    # FoBAR = np.nan(shape =[nl,ns])
    A = 1.00014
    B = 0.01671
    C = 0.9856002831
    D = 3.452868
    E = 360.
    F1 = 1./((A-B*np.cos(2.*np.pi*(C*doy-D)/E)-0.000014*np.cos(4.*np.pi*(C*doy-D)/E))**2)
    FoBAR  =F0[i]*F1
    ray_i = ray_i*FoBAR
    f.end()
    return ray_i


# def calc(rayleigh_lut_path, sza, vza, saa, vaa, windspeed, pressure, F0, bands, doy):
#     phi=saa-vaa
#     (rows, columns)=sza.shape
#     for i, _ in enumerate(rayleigh_lut_path):
#         value_ = ray_rad(press=pressure, sza=sza, vza=vza, phi=phi, wavelength=bands[i], ns=columns, nl=rows, doy=doy,
#                 INF=rayleigh_lut_path, F0=F0, lambdas=bands)
#         if i ==0:
#             ray=value_
#         else:
#             ray=np.dstack([ray,value_])
#     return ray

# def ray_rad3(raylut_info, press, sza, vza, phi, wavelength, ns, nl, doy, F0, lambdas, windspeed):
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


def ray_rad2(rayleigh_lut_path, press, sza, vza, phi, wavelength, ns, nl, doy, F0, lambdas, windspeed):
    # lut = r'C:\Users\lwk15\Documents\Tencent Files\2939027854\FileRecv/rayleigh_modisa_443_iqu.hdf'

    phi = phi + 180.
    phi_idx = np.where(phi < 180)
    phi[phi_idx] = phi[phi_idx] + 360
    phi_idx = np.where(phi > 180)
    phi[phi_idx] = phi[phi_idx] - 360

    press0 = 1013.25

    lut_file = glob.glob(rayleigh_lut_path + os.sep + 'rayleigh_*' + str(int(wavelength)) + '_iqu.hdf')[0]
    f = SD(lut_file, SDC.READ)
    sun_j = (f.select('solz')).get()
    i_ray = (f.select('i_ray')).get()  # 瑞利散射的I成分
    tau_i = (f.select('taur')).get()
    ray_ang = (f.select('senz')).get()
    try:
        ray_sigma = (f.select('sigma')).get()
    except:
        _ = (f.select('wind')).get()
        ray_sigma = 0.0731 * np.sqrt(_)
    msun = nsun_ray = 45  # 45个太阳天顶角
    nrad_ray = 41  # 41个卫星天顶角
    norder_ray = 3
    NWIND = 8

    sigma_m = 0.0731 * np.sqrt(windspeed)
    sigma_m = sigma_m.reshape(nl, ns, 1)
    sza = sza.reshape(nl, ns, 1)
    vza = vza.reshape(nl, ns, 1)
    # while sigma_m > ray_sigma[isigma2] & isigma2 < NWIND:
    #     isigma2 = isigma2 + 1

    # isun_low = np.int16(np.trunc(sza / 2))
    # isun_high = isun_low + 1

    _ = sza - sun_j
    _[_ < 0] = 999
    isun_high = np.nanargmin(_, axis=2)
    isun_high[isun_high > 44] = 44
    isun_low = isun_high - 1
    isun_low[isun_low < 0] = 0

    # j1 = np.zeros(shape=[nl, ns], dtype=np.int16)
    # j2 = np.zeros(shape=[nl, ns], dtype=np.int16)
    # for k in range(nl):
    #     for n in range(ns):
    #         j = np.where(ray_ang >= vza[k, n])
    #         j2[k, n] = j[0][0]
    #         j1[k, n] = j2[k, n] - 1

    _ = vza - ray_ang
    _[_ < 0] = 999
    j2 = np.nanargmin(_, axis=2)
    j2[j2 > 40] = 40
    j1 = j2 - 1
    j1[j1 < 0] = 0

    raj1 = ray_ang[j1]
    raj2 = ray_ang[j2]

    rsh = sun_j[isun_high]
    rsl = sun_j[isun_low]
    hsun = rsh - rsl
    hview = raj2 - raj1
    p = (sza.reshape(nl,ns) - rsl) / hsun
    q = (vza.reshape(nl,ns)  - raj1) / hview

    _ = sigma_m - ray_sigma
    _[_ < 0] = 999
    isigma2 = np.nanargmin(_, axis=2)
    isigma2[isigma2 > 10] = 10
    isigma1 = isigma2 - 1
    isigma1[isigma1 < 0] = 0
    ray_i_sig = []
    for i, isigma_ in enumerate([isigma1, isigma2]):
        ray_for_i0 = np.empty(shape=[nl,ns, nrad_ray, nsun_ray])
        ray_for_i1 = np.empty_like(ray_for_i0)
        ray_for_i2 = np.empty_like(ray_for_i0)
        for j in range(msun):
            ray_for_i0[:, :, :, j] = i_ray[isigma_, j, 0, :]
            ray_for_i1[:, :, :, j] = i_ray[isigma_, j, 1, :]
            ray_for_i2[:, :, :, j] = i_ray[isigma_, j, 2, :]
        ray_i = 0
        for m, ray_for_i in enumerate([ray_for_i0, ray_for_i1, ray_for_i2]):
            # marr = np.zeros(shape=[nl, ns], dtype=np.int16) + m
            x_range = range(0, ns)
            y_range = range(0, nl)
            x, y = np.meshgrid(x_range, y_range)

            # f00 = ray_for_i[:, :, j1, isun_low]
            # f10 = ray_for_i[:, :, j1, isun_high]
            # f01 = ray_for_i[:, :, j2, isun_low]
            # f11 = ray_for_i[:, :, j2, isun_high]
            f00 = interp(y, x, j1, isun_low, ray_for_i)
            f10 = interp(y, x, j1, isun_high, ray_for_i)
            f01 = interp(y, x, j2, isun_low, ray_for_i)
            f11 = interp(y, x, j2, isun_high, ray_for_i)
            value = (1. - p) * (1. - q) * f00 + p * q * f11 + p * (1. - q) * f10 + q * (1. - p) * f01
            ray_i = (value * np.cos(2 * np.pi * (phi * m) / 360)) + ray_i
        ray_i_sig.append(ray_i)
    h = (sigma_m.reshape(nl, ns) - ray_sigma[isigma1]) / (ray_sigma[isigma2] - ray_sigma[isigma1])
    ray_i = ray_i_sig[0] + (ray_i_sig[1] - ray_i_sig[0]) * h

    # pp0 = np.nan(shape =[nl,ns])
    # fac = np.nan(shape =[nl,ns])
    tauray = np.empty(shape=[nl, ns])
    pp0 = press / press0
    pplt0 = np.where(pp0 < 0.)
    pp0[pplt0] = 1.
    tauray[:, :] = pp0[:, :] * tau_i
    fac = (1. - np.exp(-tauray / np.cos(np.pi * vza.reshape(nl,ns) / 180))) / (1. - np.exp((-tau_i / np.cos(np.pi * vza.reshape(nl,ns) / 180))))
    ray_i = ray_i * fac

    i = np.where(lambdas == wavelength)
    # FoBAR = np.nan(shape =[nl,ns])
    A = 1.00014
    B = 0.01671
    C = 0.9856002831
    D = 3.452868
    E = 360.
    F1 = 1. / ((A - B * np.cos(2. * np.pi * (C * doy - D) / E) - 0.000014 * np.cos(
        4. * np.pi * (C * doy - D) / E)) ** 2)
    FoBAR = F0[i] * F1
    ray_i = ray_i * FoBAR
    f.end()
    return ray_i


def ray_rad_优化插值(rayleigh_lut_path, press, sza, vza, phi, wavelength, ns, nl, doy, F0, lambdas, windspeed):
    # lut = r'C:\Users\lwk15\Documents\Tencent Files\2939027854\FileRecv/rayleigh_modisa_443_iqu.hdf'

    phi = phi + 180.
    phi_idx = np.where(phi < 180)
    phi[phi_idx] = phi[phi_idx] + 360
    phi_idx = np.where(phi > 180)
    phi[phi_idx] = phi[phi_idx] - 360

    press0 = 1013.25
    lut_file = glob.glob(rayleigh_lut_path + os.sep + 'rayleigh_*' + str(int(wavelength)) + '_iqu.hdf')[0]
    f = SD(lut_file, SDC.READ)
    sun_j = (f.select('solz')).get()
    i_ray = (f.select('i_ray')).get()  # 瑞利散射的I成分
    tau_i = (f.select('taur')).get()
    ray_ang = (f.select('senz')).get()
    try:
        ray_sigma = (f.select('sigma')).get()
    except:
        _ = (f.select('wind')).get()
        ray_sigma = 0.0731 * np.sqrt(_)
    msun = nsun_ray = 45  # 45个太阳天顶角
    nrad_ray = 41  # 41个卫星天顶角
    norder_ray = 3
    NWIND = 8

    sigma_m = 0.0731 * np.sqrt(windspeed)
    sigma_m = sigma_m.reshape(nl, ns, 1)
    sza = sza.reshape(nl, ns, 1)
    vza = vza.reshape(nl, ns, 1)
    # while sigma_m > ray_sigma[isigma2] & isigma2 < NWIND:
    #     isigma2 = isigma2 + 1

    # isun_low = np.int16(np.trunc(sza / 2))
    # isun_high = isun_low + 1

    _ = sza - sun_j
    _[_ < 0] = 999
    isun_high = np.nanargmin(_, axis=2)
    isun_high[isun_high > 44] = 44
    isun_low = isun_high - 1
    isun_low[isun_low < 0] = 0

    _ = vza - ray_ang
    _[_ < 0] = 999
    j2 = np.nanargmin(_, axis=2)
    j2[j2 > 40] = 40
    j1 = j2 - 1
    j1[j1 < 0] = 0

    raj1 = ray_ang[j1]
    raj2 = ray_ang[j2]

    rsh = sun_j[isun_high]
    rsl = sun_j[isun_low]
    hsun = rsh - rsl
    hview = raj2 - raj1
    p = (sza.reshape(nl,ns) - rsl) / hsun
    q = (vza.reshape(nl,ns) - raj1) / hview

    _ = sigma_m - ray_sigma
    _[_ < 0] = 999
    isigma2 = np.nanargmin(_, axis=2)
    isigma2[isigma2 > 10] = 10
    isigma1 = isigma2 - 1
    isigma1[isigma1 < 0] = 0
    ray_i_sig = []
    for i, isigma_ in enumerate([isigma1, isigma2]):
        ray_for_i0 = np.empty(shape=[nl,ns, nrad_ray, nsun_ray])
        ray_for_i1 = np.empty_like(ray_for_i0)
        ray_for_i2 = np.empty_like(ray_for_i0)
        for j in range(msun):
            ray_for_i0[:, :, :, j] = i_ray[isigma_, j, 0, :]
            ray_for_i1[:, :, :, j] = i_ray[isigma_, j, 1, :]
            ray_for_i2[:, :, :, j] = i_ray[isigma_, j, 2, :]
        ray_i = 0
        for m, ray_for_i in enumerate([ray_for_i0, ray_for_i1, ray_for_i2]):
            # marr = np.zeros(shape=[nl, ns], dtype=np.int16) + m
            x_range = range(0, ns)
            y_range = range(0, nl)
            x, y = np.meshgrid(x_range, y_range)

            # f00 = ray_for_i[:, :, j1, isun_low]
            # f10 = ray_for_i[:, :, j1, isun_high]
            # f01 = ray_for_i[:, :, j2, isun_low]
            # f11 = ray_for_i[:, :, j2, isun_high]
            f00 = interp(y, x, j1, isun_low, ray_for_i)
            f10 = interp(y, x, j1, isun_high, ray_for_i)
            f01 = interp(y, x, j2, isun_low, ray_for_i)
            f11 = interp(y, x, j2, isun_high, ray_for_i)
            value = (1. - p) * (1. - q) * f00 + p * q * f11 + p * (1. - q) * f10 + q * (1. - p) * f01
            ray_i = (value * np.cos(2 * np.pi * (phi * m) / 360)) + ray_i
        ray_i_sig.append(ray_i)
    h = (sigma_m.reshape(nl, ns) - ray_sigma[isigma1]) / (ray_sigma[isigma2] - ray_sigma[isigma1])
    ray_i = ray_i_sig[0] + (ray_i_sig[1] - ray_i_sig[0]) * h

    # pp0 = np.nan(shape =[nl,ns])
    # fac = np.nan(shape =[nl,ns])
    tauray = np.empty(shape=[nl, ns])
    pp0 = press / press0
    pplt0 = np.where(pp0 < 0.)
    pp0[pplt0] = 1.
    tauray[:, :] = pp0[:, :] * tau_i
    fac = (1. - np.exp(-tauray / np.cos(np.pi * vza.reshape(nl,ns) / 180))) / (1. - np.exp((-tau_i / np.cos(np.pi * vza.reshape(nl,ns) / 180))))
    ray_i = ray_i * fac

    i = np.where(lambdas == wavelength)
    # FoBAR = np.nan(shape =[nl,ns])
    A = 1.00014
    B = 0.01671
    C = 0.9856002831
    D = 3.452868
    E = 360.
    F1 = 1. / ((A - B * np.cos(2. * np.pi * (C * doy - D) / E) - 0.000014 * np.cos(
        4. * np.pi * (C * doy - D) / E)) ** 2)
    FoBAR = F0[i] * F1
    ray_i = ray_i * FoBAR
    f.end()
    return ray_i


def interp(y,x,ind3,ind4, mat):
    index_matrix_dim1 = y[:, :, np.newaxis, np.newaxis]  # 第一个维度的索引
    index_matrix_dim2 = x[:, :, np.newaxis, np.newaxis]  # 第二个维度的索引
    index_matrix_dim3 = ind3[:, :, np.newaxis, np.newaxis]  # 第三个维度的索引
    index_matrix_dim4 = ind4[:, :, np.newaxis, np.newaxis]
    # f = np.empty(x.shape)
    # # 使用循环逐个索引获取值
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         indices = (index_matrix_dim1[i, j, 0, 0], index_matrix_dim2[i, j, 0, 0], index_matrix_dim3[i, j, 0, 0],
    #                    index_matrix_dim4[i, j, 0, 0])
    #         # try:
    #         f[i, j] = mat[indices]

    f = np.take(mat, np.ravel_multi_index((index_matrix_dim1.flatten(),
                                           index_matrix_dim2.flatten(),
                                           index_matrix_dim3.flatten(),
                                           index_matrix_dim4.flatten()),
                                          dims=mat.shape))
    f = f.reshape(index_matrix_dim1.shape)
    f=f[:,:,0,0]
    return f

# def rayleigh(rayleigh_lut_path, windspeed, bands:list,sza,saa,vza,vaa):
#     reaa=saa-vaa
#     lut_file = glob.glob(rayleigh_lut_path + os.sep + 'rayleigh_*' + str(int(wavelength)) + '_iqu.hdf')[0]
#     f = SD(lut_file, SDC.READ)
#     # // read the variables
#     ray_sol = (f.select('solz')).get()
#     ray_for_i = (f.select('i_ray')).get()  # 瑞利散射的I成分
#     tau_i = (f.select('taur')).get()
#     ray_sen = (f.select('senz')).get()
#     ray_sigma = (f.select('sigma')).get()
#     # // check to make sure the dimensions are correct, so we don't overwrite memory
#     # check_dimension_size(file, nc_id, "nrad_ray", NSEN);
#     # check_dimension_size(file, nc_id, "nsun_ray", NSOL);
#     # check_dimension_size(file, nc_id, "nwind_ray", NWIND);
#     # check_dimension_size(file, nc_id, "norder_ray", NORDER);
#     NSEN=ray_sen.size()
#     NSOL=ray_sol.size()
#     NWIND=ray_sigma.size()
#     NORDER = 3
#
#     sigma_m = 0.0731 * np.sqrt(windspeed)
#     isigma2 = 0
#     while sigma_m > ray_sigma[isigma2] & isigma2 < NWIND:
#         isigma2 = isigma2 + 1
#     isigma1 = isigma2 - 1
#     l1rec = {}
#     l1rec["geom_per_band"] = None
#     nwave=bands.size
#     if l1rec.geom_per_band is not None:
#         gmult = 1
#     else:
#         gmult = 0
#     r_phi=reaa
#     r_solz=sza
#     r_senz=vza
#     for iw in range(nwave):
#         ray_i = 0.0
#         ray_q = 0.0
#         ray_u = 0.0
#         ix = iw if l1rec.geom_per_band is None else iw * gmult
#         cosd_phi = [np.cos(np.deg2rad(r_phi[ix]) * m) for m in range(NORDER)]
#         sind_phi = [np.sin(np.deg2rad(r_phi[ix]) * m) for m in range(NORDER)]
#         isol1 = int(r_solz[ix]) // 2   ##??????????
#         isol2 = isol1 + 1
#         for isen2 in range(NSEN):
#             if r_senz[ix] < ray_sen[isen2]:
#                 break
#         isen1 = isen2 - 1
#
#         if isol1 >= NSOL - 1:
#             isol1 = NSOL - 1
#             isol2 = isol1
#             p = 1.0
#         else:
#             p = (r_solz[ix] - ray_sol[isol1]) / (ray_sol[isol2] - ray_sol[isol1])
#         q = (r_senz[ix] - ray_sen[isen1]) / (ray_sen[isen2] - ray_sen[isen1])
#
#         airmass = 1.0 / np.cos(np.deg2rad(r_solz[ix])) + 1.0 / np.cos(np.deg2rad(r_senz[ix]))
#
#         ray_i_sig = []
#         ray_q_sig = []
#         ray_u_sig = []
#         for isigma in range(isigma1, isigma2 + 1):
#             iwind = isigma - isigma1  # 0 or 1
#             ray_i_sig[iwind] = 0.0
#             ray_q_sig[iwind] = 0.0
#             ray_u_sig[iwind] = 0.0
#             # I component
#             for m in range(NORDER):
#                 f00 = ray_for_i[iw][isigma][isol1][m][isen1]
#                 f10 = ray_for_i[iw][isigma][isol2][m][isen1]
#                 f01 = ray_for_i[iw][isigma][isol1][m][isen2]
#                 f11 = ray_for_i[iw][isigma][isol2][m][isen2]
#
#                 ray_i_sig[iwind] += ((1.0 - p) * (1.0 - q) * f00 + p * q * f11 + p * (1.0 - q) * f10 + q * (
#                             1.0 - p) * f01) * cosd_phi[m]


if __name__ == '__main__':
    press = np.array([[1013, 1013]])
    sza = np.array([[10, 70]])
    vza = np.array([[20, 60]])
    phi = np.array([[0, 140]])
    wavelength = 442
    ns = 2
    nl = 1
    doy = 100
    rc = ray_rad(press, sza, vza, phi, wavelength, ns, nl, doy)
    print(rc)