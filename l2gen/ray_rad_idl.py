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


def ray_rad(rayleigh_lut_path, press, sza, vza, phi, wavelength, ns, nl, doy,F0, lambdas):
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


def calc(rayleigh_lut_path, sza, vza, saa, vaa, windspeed, pressure, F0, bands, doy):
    phi=saa-vaa
    (rows, columns)=sza.shape
    for i, _ in enumerate(rayleigh_lut_path):
        value_ = ray_rad(press=pressure, sza=sza, vza=vza, phi=phi, wavelength=bands[i], ns=columns, nl=rows, doy=doy,
                INF=rayleigh_lut_path, F0=F0, lambdas=bands)
        if i ==0:
            ray=value_
        else:
            ray=np.dstack([ray,value_])
    return ray



if __name__ == '__main__':

    press = np.array([[1013, 1013]])
    sza = np.array([[10,70]])
    vza = np.array([[20,60]])
    phi = np.array([[0,140]])
    wavelength = 442
    ns = 2
    nl = 1
    doy = 100
    rc = ray_rad(press, sza, vza, phi, wavelength, ns, nl, doy)
    print(rc)