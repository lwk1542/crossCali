# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: brdf111111.py
@time: 2021/5/22 9:41
@desc:
"""

# /* ---------------------------------------------------------------------------- */
# /* foqint_morel() - reads and interpolates f/Q tables of Morel & Gentilli       */
# /*                                                                              */
# /* wave[] - list of input wavelengths (nm)                                      */
# /* nwave  - number of input wavelengths                                         */
# /* solz   - solar zenith angle of observation (deg)                             */
# /* senzp  - view  zenith angle of observation, below surface (deg)              */
# /* phi    - relative azimuth of observation, 0=<--> (deg)                       */
# /* chl    - chlorophyll-a concentration (mg/m^3)                                */
# /* brdf[] - band-indexed array of f/Q corrections per wavelength (f0/Q0)/(f/Q)  */
# /*                                                                              */
#
# /* ---------------------------------------------------------------------------- */

import h5py
import numpy as np
from scipy import interpolate
import get_chl
import sys
sys.path.append("..")
from sharepy import predefine


class BRDF:
    def __init__(self, vza: np.ndarray = None, sza: np.ndarray = None, vaa: np.ndarray = None,
                 saa: np.ndarray = None, bands: np.ndarray = None, F0: np.ndarray = None,
                 chl: np.ndarray = None, nlw: np.ndarray = None, b443: int = None, b490: int = None,
                 b520: int = None, b555: int = None, b670: int = None, foqopt: str = "FOQMOREL", ws: int = None,
                 fqfile=r'C:\git_repository\liwenkai\atmosphericCorrection\LUT/morel_fq.h5'):
        self.vza = vza
        self.sza = sza
        self.vaa = vaa
        self.saa = saa
        self.bands = bands
        self.F0 = F0
        self.chl = chl
        self.nlw = nlw
        self.b443 = b443
        self.b490 = b490
        self.b520 = b520
        self.b555 = b555
        self.b670 = b670
        self.foqopt = foqopt
        self.fqfile = fqfile
        self.ws = ws
        self.ws[self.ws < 0] = 0
        self.ws[self.ws > 30] = 30

    def ocbrdf(self):

        self.reaa = self.vaa - 180 - self.saa
        self.reaa[self.reaa < -180] = self.reaa[self.reaa < -180] + 360
        self.reaa[self.reaa > 180] = self.reaa[self.reaa > 180] - 360
        # 1 初始化
        brdf = np.ones(shape=(self.sza.shape[0], self.sza.shape[1], self.bands.size))

        # 2 /* transmittance of view path through air & sea interface */
        tf = self.fresnel_sen(return_tf=1)

        # 3 /* transmittance of solar path through air & sea interface */
        temp = self.fresnel_sol(return_tf=1)

        brdf = brdf * tf.reshape(tf.shape[0], tf.shape[1], 1) * temp

        #  4 /* Morel f/Q correction */
        if self.foqopt == "FOQMOREL":
            brdf = brdf * self.foq_morel()

        #  5 /* Gordon correction of diffuse transmittance */ 暂时不加这一步，seadas默认算法没有这一步
        # dtran_brdf(l2rec, ip, wave, nwave, Fo, nLw, chl, temp)

        return brdf

    def fresnel_sen(self, return_tf=0):
        # /* ---------------------------------------------------------------------------- */
        # /* fresnel_sen() - effects of the air-sea transmittance for sensor view         */
        # /*                                                                              */
        # /* Description:                                                                 */
        # /*   This computes effects of the air-sea transmittance (depending on sensor    */
        # /*   zenith angle) on the derived normalized water-leaving radiance.            */
        # /*   Menghua Wang 5/27/02.                                                      */
        # /*                                                                              */
        # /* modified to return fresnel transmittance as option, December 2008, BAF       */
        #
        # /* ---------------------------------------------------------------------------- */
        tf0 = 0.9795218
        nw = 1.334
        mu = np.cos(self.vza * np.pi / 180)

        sq = np.sqrt(nw * nw - 1. + mu * mu)
        r2 = np.power((mu - sq) / (mu + sq), 2)
        q1 = (1. - mu * mu - mu * sq) / (1. - mu * mu + mu * sq)
        fres = r2 * (q1 * q1 + 1.) / 2.0
        tf = 1. - fres
        brdf = tf0 / tf

        if return_tf != 0:
            return tf
        else:
            return brdf

    def fresnel_sol(self, return_tf=0):
        # /* ---------------------------------------------------------------------------- */
        # /* fresnel_sol() - effects of the air-sea transmittance for solar path          */
        # /*                                                                              */
        # /* Description:                                                                 */
        # /*   This computes the correction factor on normalized water-leaving radiance   */
        # /*   to account for the solar zenith angle effects on the downward irradiance   */
        # /*   from above the ocean surface to below the surface.                         */
        # /*   Menghua Wang 9/27/04.                                                      */
        # /*                                                                              */
        # /* Added windspeed dependence, December 2004, BAF                               */
        # /* Modified to return air-sea transmittance as option, December 2008, BAF       */
        #
        # /* ---------------------------------------------------------------------------- */
        twave = np.array([412., 443., 490., 510., 555., 670.])
        tsigma = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        # /* M Wang, personal communication, red-nir iterpolated */
        tf0_w = [412., 443., 490., 510., 555., 670., 765., 865.]
        tf0_v = [0.965980, 0.968320, 0.971040, 0.971860, 0.973450, 0.977513, 0.980870, 0.984403]

        ws_inter = [0.0, 1.9, 7.5, 16.9, 30]
        # c在不同风速下的abcd值矩阵
        # c=np.array([
        #     [
        #         [-0.0087, -0.0122, -0.0156, -0.0163, -0.0172, -0.0172],
        #         [0.0638, 0.0415, 0.0188, 0.0133, 0.0048, -0.0003],
        #         [-0.0379, -0.0780, -0.1156, -0.1244, -0.1368, -0.1430],
        #         [-0.0311, -0.0427, -0.0511, -0.0523, -0.0526, -0.0478],
        #     ],
        #     [
        #         [-0.0011, -0.0037, -0.0068, -0.0077, -0.0090, -0.0106],
        #         [0.0926, 0.0746, 0.0534, 0.0473, 0.0368, 0.0237],
        #         [-5.3E-4, -0.0371, -0.0762, -0.0869, -0.1048, -0.1260],
        #         [-0.0205, -0.0325, -0.0438, -0.0465, -0.0506, -0.0541],
        #     ],
        #     [
        #         [6.8E-5, -0.0018, -0.0011, -0.0012, -0.0015, -0.0013],
        #         [0.1150, 0.1115, 0.1075, 0.1064, 0.1044, 0.1029],
        #         [0.0649, 0.0379, 0.0342, 0.0301, 0.0232, 0.0158],
        #         [0.0065, -0.0039, -0.0036, -0.0047, -0.0062, -0.0072],
        #     ],
        #     [
        #         [-0.0088, -0.0097, -0.0104, -0.0106, -0.0110, -0.0111],
        #         [0.0697, 0.0678, 0.0657, 0.0651, 0.0640, 0.0637],
        #         [0.0424, 0.0328, 0.0233, 0.0208, 0.0166, 0.0125],
        #         [0.0047, 0.0013, -0.0016, -0.0022, -0.0031, -0.0036],
        #     ],
        #     [
        #         [-0.0081, -0.0089, -0.0096, -0.0098, -0.0101, -0.0104],
        #         [0.0482, 0.0466, 0.0450, 0.0444, 0.0439, 0.0434],
        #         [0.0290, 0.0220, 0.0150, 0.0131, 0.0103, 0.0070],
        #         [0.0029, 0.0004, -0.0017, -0.0022, -0.0029, -0.0033],
        #     ]
        # ])

        a = np.array([
            [-0.0087, -0.0122, -0.0156, -0.0163, -0.0172, -0.0172],
            [-0.0011, -0.0037, -0.0068, -0.0077, -0.0090, -0.0106],
            [6.8E-5, -0.0018, -0.0011, -0.0012, -0.0015, -0.0013],
            [-0.0088, -0.0097, -0.0104, -0.0106, -0.0110, -0.0111],
            [-0.0081, -0.0089, -0.0096, -0.0098, -0.0101, -0.0104],
        ])
        b = np.array([
            [0.0638, 0.0415, 0.0188, 0.0133, 0.0048, -0.0003],
            [0.0926, 0.0746, 0.0534, 0.0473, 0.0368, 0.0237],
            [0.1150, 0.1115, 0.1075, 0.1064, 0.1044, 0.1029],
            [0.0697, 0.0678, 0.0657, 0.0651, 0.0640, 0.0637],
            [0.0482, 0.0466, 0.0450, 0.0444, 0.0439, 0.0434]
        ])
        c = np.array([
            [-0.0379, -0.0780, -0.1156, -0.1244, -0.1368, -0.1430],
            [-5.3E-4, -0.0371, -0.0762, -0.0869, -0.1048, -0.1260],
            [0.0649, 0.0379, 0.0342, 0.0301, 0.0232, 0.0158],
            [0.0424, 0.0328, 0.0233, 0.0208, 0.0166, 0.0125],
            [0.0290, 0.0220, 0.0150, 0.0131, 0.0103, 0.0070],
        ])
        d = np.array([
            [-0.0311, -0.0427, -0.0511, -0.0523, -0.0526, -0.0478],
            [-0.0205, -0.0325, -0.0438, -0.0465, -0.0506, -0.0541],
            [0.0065, -0.0039, -0.0036, -0.0047, -0.0062, -0.0072],
            [0.0047, 0.0013, -0.0016, -0.0022, -0.0031, -0.0036],
            [0.0029, 0.0004, -0.0017, -0.0022, -0.0029, -0.0033]
        ])

        self.ws[self.ws < 0] = 0
        sigma = 0.0731 * np.sqrt(self.ws)
        self.sza[self.sza > 80] = 80
        x = np.log(np.cos(self.sza * np.pi / 180))
        x2 = x * x
        x3 = x * x2
        x4 = x * x3
        brdf_value = np.ones(shape=(sigma.shape[0], sigma.shape[1], self.bands.size))
        for ib in range(self.bands.size):
            band_mat = np.full_like(sigma, fill_value=self.bands[ib])
            a_interp = interpolate.interpn((tsigma.reshape(-1), twave.reshape(-1)), a,
                                           np.stack([sigma, band_mat], axis=2),
                                           method='linear', fill_value=False, bounds_error=False)
            b_interp = interpolate.interpn((tsigma.reshape(-1), twave.reshape(-1)), b,
                                           np.stack([sigma, band_mat], axis=2),
                                           method='linear', fill_value=False, bounds_error=False)
            c_interp = interpolate.interpn((tsigma.reshape(-1), twave.reshape(-1)), c,
                                           np.stack([sigma, band_mat], axis=2),
                                           method='linear', fill_value=False, bounds_error=False)
            d_interp = interpolate.interpn((tsigma.reshape(-1), twave.reshape(-1)), d,
                                           np.stack([sigma, band_mat], axis=2),
                                           method='linear', fill_value=False, bounds_error=False)
            brdf_value[:, :, ib] = 1. + a_interp * x + b_interp * x2 + c_interp * x3 + d_interp * x4

        if return_tf != 0:
            func = interpolate.interp1d(tf0_w, tf0_v, kind='linear', fill_value="extrapolate", bounds_error=False)
            tf0 = func(self.bands)
            brdf = tf0.reshape(1, 1, -1) / brdf_value
        return brdf

    def foqint_morel(self, sza: np.ndarray = None, vzap: np.ndarray = None, reaa: np.ndarray = None):
        fq = h5py.File(self.fqfile, 'r')
        lchl_lut = np.log(fq['chl'][()])
        phi_lut = fq['phi'][()]
        senz_lut = fq['senz'][()]
        solz_lut = fq['solz'][()]
        wave_lut = fq['wave'][()]
        foq_lut = fq['foq'][()]
        self.chl[self.chl < 0.01] = 0.01
        lchl = np.log(self.chl)
        vzap[vzap < np.nanmin(senz_lut)] = np.nanmin(senz_lut)
        vzap[vzap > np.nanmax(senz_lut)] = np.nanmax(senz_lut)
        fq = np.full(shape=(self.sza.shape[0], self.sza.shape[1], self.bands.size), fill_value=np.nan)
        for ib in range(self.bands.size):
            band_mat = np.full_like(self.sza, fill_value=self.bands[ib])
            fq[:, :, ib] = interpolate.interpn(
                (wave_lut.reshape(-1), solz_lut.reshape(-1), lchl_lut.reshape(-1), senz_lut.reshape(-1),
                 phi_lut.reshape(-1)), foq_lut,
                np.stack([band_mat, sza, lchl, vzap, reaa], axis=2), method='linear', bounds_error=False,
                fill_value=None)
        return fq

    def foq_morel(self):
        """
        senz: np.ndarray[float] = None, solz: np.ndarray[float] = None, vaa: np.ndarray[float] = None,
                  saa: np.ndarray[float] = None, bands: np.ndarray[float] = None, F0: np.ndarray[float] = None,
                  chl: np.ndarray[float] = None, nlw: np.ndarray[float] = None, b443:int=None, b490:int=None,
                  b520:int=None,b555:int=None,b670:int=None, foqopt:str="QMOREL",
                  fqfile=r'C:\git_repository\liwenkai\atmosphericCorrection\LUT/morel_fq.h5'
        Returns:

        """
        """
        /* ---------------------------------------------------------------------------- */
        /* foq_morel() - computes f/Q correction of Morel & Gentilli by iteration       */
        /*                                                                              */
        /* foqopt - 0=full f/Q, 1=no normalization to sun overhead (fixed f)            */
        /* sensorID - MSl12 sensor identification number                                */
        /* wave[] - list of input wavelengths (nm)                                      */
        /* nwave  - number of input wavelengths                                         */
        /* nLw[]  - normalize water-leaving radiances per wave (mW/cm^2/um/sr)          */
        /* Fo[]   - solar irradiance per wave (mW/cm^2/um/sr)                           */
        /* solz   - solar zenith angle of observation (deg)                             */
        /* senzp  - view  zenith angle of observation, below surface (deg)              */
        /* phi    - relative azimuth of observation, 0=<--> (deg)                       */
        /* brdf[] - band-indexed array of f/Q corrections per wavelength (f0/Q0)/(f/Q)  */
    
        /* ---------------------------------------------------------------------------- */
        Returns:
            computes f/Q correction of Morel & Gentilli by iteration
    
        """
        # print("foq morel")
        phip = np.abs(self.reaa)
        nw = 1.334
        senzp = np.arcsin(np.sin(self.vza * np.pi / 180.) / nw) * 180. / np.pi
        rrs = self.nlw / self.F0
        # 1. /* Compute starting chlorophyll (if not supplied) */
        chl = get_chl.get_default_chl(rrs=rrs, bands=self.bands, b443=self.b443, b490=self.b490, b520=self.b520,
                                      b555=self.b555,
                                      b670=self.b670)

        # 2. mask 可能出现叶绿素小于0 的情况，这种需要掩膜掉
        mask = chl * 1.
        mask[np.isnan(mask)] = -999
        mask[mask < 0] = np.nan
        mask2 = chl * 1.

        # 3.迭代计算brdf和叶绿素
        maxiter = predefine.thresholds().brdf_maxiter
        compchl = 1
        for iter in range(maxiter):
            # print("迭代计算气溶胶，第{0}次".format(iter))
            if self.foqopt == "QMOREL":
                foq0 = self.foqint_morel(sza=self.sza, vzap=np.zeros_like(self.sza), reaa=np.zeros_like(self.sza))
            else:
                foq0 = self.foqint_morel(sza=np.zeros_like(self.sza), vzap=np.zeros_like(self.sza),
                                         reaa=np.zeros_like(self.sza))
            foq = self.foqint_morel(sza=self.sza, vzap=senzp, reaa=phip)

            brdf = foq0 / foq
            brdf[chl < 0] = 1.
            rrs = self.nlw * brdf / self.F0

            chl = get_chl.get_default_chl(rrs=rrs, bands=self.bands, b443=self.b443, b490=self.b490, b520=self.b520,
                                          b555=self.b555, b670=self.b670)
        brdf[np.isnan(mask)] = 1.
        brdf[np.isnan(mask2)] = np.nan

        return brdf

    def dtran_brdf(self):
        modindex = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
        wavetab = [412., 443., 490., 510., 555., 670., 765., 865.]

        #  /* want relative azimuth in radians, other angles in degrees */
        phi = self.reaa * np.pi / 180.

        # * Compute starting chlorophyll (if not supplied) */
        #     if (chl < 0.0) {
        #         for (iw = 0; iw < nwave; iw++) {
        #             Rrs[iw] = nLw[iw] / Fo[iw];
        #         }
        #         chl = get_default_chl(l2rec, Rrs);
        #     }


def foqint_morel(fqfile=r'C:\git_repository\liwenkai\atmosphericCorrection\LUT/morel_fq.h5', wave=None,
                 sza=None, vzap: np.ndarray = None, reaa=None, chl=None):
    fq = h5py.File(fqfile, 'r')
    lchl_lut = np.log(fq['chl'][()])
    phi_lut = fq['phi'][()]
    senz_lut = fq['senz'][()]
    solz_lut = fq['solz'][()]
    wave_lut = fq['wave'][()]
    foq_lut = fq['foq'][()]

    chl[chl < 0.01] = 0.01
    lchl = np.log(chl)
    vzap[vzap < np.nanmin(senz_lut)] = np.nanmin(senz_lut)
    vzap[vzap > np.nanmax(senz_lut)] = np.nanmax(senz_lut)
    fq = np.full(shape=(sza.shape[0], sza.shape[1], wave.size), fill_value=np.nan)
    for ib in range(wave.size):
        band = np.full_like(sza, fill_value=wave[ib])
        fq[:, :, ib] = interpolate.interpn((
            wave_lut.reshape(-1), solz_lut.reshape(-1), lchl_lut.reshape(-1), senz_lut.reshape(-1),
            phi_lut.reshape(-1)),
            foq_lut, np.stack([band, sza, lchl, vzap, reaa], axis=2), method='linear',
            bounds_error=False, fill_value=None)

    return fq
