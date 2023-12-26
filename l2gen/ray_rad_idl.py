# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/22 10:11
@FileName: ray_rad.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""

import numpy as np
from scipy import interpolate


def ray_rad(rayleigh_infos, press, sza, vza,saa,vaa, doy,F0, bands, windspeed):
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
    phi=saa - vaa
    phi = phi + 180.
    phi_idx = np.where(phi < 180)
    phi[phi_idx] = phi[phi_idx] + 360
    phi_idx = np.where(phi > 180)
    phi[phi_idx] = phi[phi_idx] - 360

    press0 = 1013.25

    Lr_i = np.zeros(shape=(sza.shape[0], sza.shape[1], bands.size))
    for i, band in enumerate(bands):
        raylut_info = rayleigh_infos[str(int(band))]
        tau_i = raylut_info["taur"]
        ray_ang = raylut_info["senz"]
        sun_j = raylut_info["solz"]
        sigma = raylut_info["sigma"]
        i_ray = raylut_info["i_ray"]

        Norder0 = np.zeros_like(sza)
        ray_i0 = interpolate.interpn((sigma.reshape(-1), sun_j.reshape(-1), np.arange(3).reshape(-1), ray_ang.reshape(-1)), i_ray,
                    np.stack([windspeed*0., sza, Norder0, vza], axis=2), method='linear')
        ray_i1 = interpolate.interpn(
            (sigma.reshape(-1), sun_j.reshape(-1), np.arange(3).reshape(-1), ray_ang.reshape(-1)), i_ray,
            np.stack([windspeed * 0., sza, Norder0+1, vza], axis=2), method='linear')
        ray_i2 = interpolate.interpn(
            (sigma.reshape(-1), sun_j.reshape(-1), np.arange(3).reshape(-1), ray_ang.reshape(-1)), i_ray,
            np.stack([windspeed * 0., sza, Norder0+2, vza], axis=2), method='linear')
        ray_i_temp = ray_i0 * np.cos(2 * np.pi * (phi * 0) / 360) + ray_i1 * np.cos(
            2 * np.pi * (phi * 1) / 360) + ray_i2 * np.cos(2 * np.pi * (phi * 2) / 360)
        ray_i=ray_i_temp
        # pp0 = np.nan(shape =[nl,ns])
        # fac = np.nan(shape =[nl,ns])
        tauray = np.empty_like(sza)
        pp0 = press/press0
        pplt0 = np.where(pp0 < 0.)
        pp0[pplt0] = 1.
        tauray[:, :] = pp0[:, :]*tau_i
        fac = (1.-np.exp(-tauray/np.cos(np.pi*vza/180)))/(1.-np.exp((-tau_i/np.cos(np.pi*vza/180))))
        ray_i = ray_i*fac

        # i = np.where(lambdas == wavelength)
        # FoBAR = np.nan(shape =[nl,ns])
        A = 1.00014
        B = 0.01671
        C = 0.9856002831
        D = 3.452868
        E = 360.
        F1 = 1./((A-B*np.cos(2.*np.pi*(C*doy-D)/E)-0.000014*np.cos(4.*np.pi*(C*doy-D)/E))**2)
        FoBAR = F0[i] * F1
        ray_i = ray_i * FoBAR
        Lr_i[:, :, i] = ray_i

    return Lr_i


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