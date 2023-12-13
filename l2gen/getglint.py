# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: getglint.py
@time: 2021/3/12 16:26
@desc:
https://oceancolor.gsfc.nasa.gov/docs/ocssw/glint_8c_source.html
"""
import numpy as np
from sharepy import predefine


class get_glint_coefficient:
    @staticmethod
    def getglint(vza=None, sza=None, vaa=None, saa=None, windspeed=None, winddirection=None):
        """
          Calculate sun gliter coefficient

           X1  Angle MU       (sensor zenith angle) VZA
           X2  Angle PHI      (solar zenith angle)  SZA
           X3  Angle NU       (sensor-sun azimuth) reaa
          X4  Wind speed     (m/s) windspeed
           X5  Wind direction (radians)  winddirection
           X6  Radiance       (ignoring the atmosphere)

        """
        reaa = vaa - saa
        reaa[reaa > 180.] = reaa[reaa > 180.] - 180

        y4 = np.max(windspeed, 0.001)

        y1 = np.deg2rad(vza)
        y1[y1 == 0] = 1.e-7
        y2 = np.deg2rad(sza)
        y2[y2 == 0] = 1.e-7
        y3 = np.deg2rad(reaa)
        omega = np.acoss(np.cos(y1) * np.cos(y2) - np.sin(y1) * np.sin(y2) * np.cos(y3)) / 2.
        omega[omega == 0] = 1.e-7
        beta = np.acoss((np.cos(y1) + np.cos(y2)) / (2. * np.cos(omega)))
        beta[beta == 0] = 1.e-7
        alpha = np.acoss((np.cos(beta) * np.cos(y2) - np.cos(omega)) / (np.sin(beta) * np.sin(y2)))
        alpha[np.sin(y3) < .0] = -alpha[np.sin(y3) < .0]

        # Isotropic wind

        #           ! from Cox & Munk

        sigc = .04964 * np.sqrt(y4)
        sigu = .04964 * np.sqrt(y4)

        chi = winddirection
        alphap = alpha + chi
        swig = np.sin(alphap) * np.tan(beta) / sigc
        eta = np.cos(alphap) * np.tan(beta) / sigu
        expon = -(swig ** 2 + eta ** 2) / 2.
        expon[expon < -30] = -30  # trap underflow
        expon[expon > 30] = 30  # trap overflow

        prob = np.exp(expon) / (2. * np.pi * sigu * sigc)
        rho = get_glint_coefficient.reflec(omega)

        # Normal distribution
        x6 = rho * prob / (4. * np.cos(y1) * np.cos(beta) ** 4)

        return x6

    @staticmethod
    def getglint_iqu(vza=None, sza=None, vaa=None, saa=None, windspeed=None, winddirection=None):
        """
          Calculate sun gliter coefficient

           X1  Angle MU       (sensor zenith angle) VZA
           X2  Angle PHI      (solar zenith angle)  SZA
           X3  Angle NU       (sensor-sun azimuth) reaa
           X4  Wind speed     (m/s) windspeed
           X5  Wind direction (radians)  winddirection

           X6(real) - Sun glitter coefficient
           X7(real) - Q/I for glitter
           X8(real) - U/I for glitter

        """
        reaa = vaa - 180 - saa
        reaa[reaa < -180] = reaa[reaa < -180] + 360
        reaa[reaa > 180] = reaa[reaa > 180] - 360

        windspeed[windspeed < 0.001] = 0.001
        y4 = windspeed
        y1 = np.deg2rad(vza)
        y1[y1 == 0] = 1.e-7
        y2 = np.deg2rad(sza)
        y2[y2 == 0] = 1.e-7
        y3 = np.deg2rad(reaa)
        omega = np.arccos(np.cos(y1) * np.cos(y2) - np.sin(y1) * np.sin(y2) * np.cos(y3)) / 2.
        omega[omega == 0] = 1.e-7
        beta = np.arccos((np.cos(y1) + np.cos(y2)) / (2. * np.cos(omega)))
        beta[beta == 0] = 1.e-7
        z = (np.cos(beta) * np.cos(y2) - np.cos(omega)) / (np.sin(beta) * np.sin(y2))
        z[z < -1] = -1
        z[z > 1] = 1
        alpha = np.arccos(z)
        alpha[np.sin(y3) < .0] = -alpha[np.sin(y3) < .0]

        # Isotropic wind

        #           ! from Cox & Munk

        sigc = .04964 * np.sqrt(y4)
        sigu = .04964 * np.sqrt(y4)

        chi = winddirection
        alphap = alpha + chi
        swig = np.sin(alphap) * np.tan(beta) / sigc
        eta = np.cos(alphap) * np.tan(beta) / sigu
        expon = -(swig ** 2 + eta ** 2) / 2.
        expon[expon < -30] = -30  # trap underflow
        expon[expon > 30] = 30  # trap overflow

        prob = np.exp(expon) / (2. * np.pi * sigu * sigc)
        # rho = reflec(omega)

        rho_plus, rho_minus = get_glint_coefficient.reflec_both(omega)

        # Normal distribution
        x6 = rho_plus * prob / (4. * np.cos(y1) * np.cos(beta) ** 4)

        # Polarization components
        cr = np.empty_like(omega)
        sr = np.empty_like(omega)
        rot_ang = np.empty_like(omega)
        loc=omega > .0001
        cr[loc] = (np.cos(y2[loc]) - np.cos(2. * omega[loc]) * np.cos(
            y1[loc])) / (np.sin(2. * omega[loc]) * np.sin(y2[loc]))
        sr[loc] = np.sin(y2[loc]) * np.sin(np.pi - y3[loc]) / np.sin(
            2. * omega[loc])
        sr[sr > 1] = 1
        sr[sr < -1] = -1
        rot_ang[loc] = np.sign(cr[loc]) * np.arcsin(
            sr[loc])  # rot_ang = sign(1., cr)*asinn(sr)
        rot_ang[~loc] = np.pi / 2
        c2r = np.cos(2. * rot_ang)
        s2r = np.sin(2. * rot_ang)

        x7 = c2r * rho_minus / rho_plus  # q_ov_i
        x8 = -s2r * rho_minus / rho_plus  # u_ov_i

        return x6, x7, x8

    @staticmethod
    def reflec(x1):
        """
       C
       C  X1  Incident angle (radians)
       C  X3  Reflectance
       C
       C       n1 sin(x1) = n2 sin(x2)
       C
       C                    tan(x1-x2)**2
       C       Refl(par ) = -------------
       C                    tan(x1+x2)**2
       C
       C                    sin(x1-x2)**2
       C       Refl(perp) = -------------
       C                    sin(x1+x2)**2
       C
       C  Where:
       C       x1  Incident angle
       C       n1  Index refraction of Air
       C       x2  Refracted angle
       C       n2  Index refraction of Water
       C
             real X1, X2, X3, ref
       *                                       ! Index refraction of sea water
       """
        ref = 4. / 3.
        x3 = np.empty_like(x1)
        x3[x1 < .00001] = .0204078

        x2 = np.asin(np.sin(x1) / ref)
        x3 = (np.sin(x1 - x2) / np.sin(x1 + x2)) ** 2 + (np.tan(x1 - x2) / np.tan(x1 + x2)) ** 2
        x3[x1 >= .00001] = x3[x1 >= .00001] / 2.

        return x3

    @staticmethod
    def reflec_both(x1):
        """
        C
        C  X1  Incident angle (radians)
        C  X3  Reflectance sum
        C  X4  Reflectance difference
        C
        C       n1 sin(x1) = n2 sin(x2)
        C
        C                    tan(x1-x2)**2
        C       Refl(par ) = -------------
        C                    tan(x1+x2)**2
        C
        C                    sin(x1-x2)**2
        C       Refl(perp) = -------------
        C                    sin(x1+x2)**2
        C
        C  Where:
        C       x1  Incident angle
        C       n1  Index refraction of Air
        C       x2  Refracted angle
        C       n2  Index refraction of Water
        C
              real X1, X2, X3, X4, REF
              real PERP, PAR
        c                                       ! Index refraction of sea water
        """
        ref = 4. / 3.
        x3 = np.empty_like(x1)
        x4 = np.empty_like(x1)
        x3[x1 < .00001] = .0204078
        x4[x1 < .00001] = 0.

        x2 = np.arcsin(np.sin(x1) / ref)
        perp = (np.sin(x1 - x2) / np.sin(x1 + x2)) ** 2
        par = (np.tan(x1 - x2) / np.tan(x1 + x2)) ** 2
        x3[x1 > .00001] = perp[x1 > .00001] + par[x1 > .00001]
        x3[x1 > .00001] = x3[x1 > .00001] / 2.
        x4[x1 > .00001] = -perp[x1 > .00001] + par[x1 > .00001]
        x4[x1 > .00001] = x4[x1 > .00001] / 2.

        return x3, x4


def taua_est(x):
    return -0.8 - 0.4 * np.log(x)


class sunglint_contamination_reflectances:
    @staticmethod
    def calcu(iter_num=None, glint_coef=None, airmass=None, sza=None, taur=None, F0=None, La=None, taua=None, mode=2):
        """
        https://oceancolor.gsfc.nasa.gov/docs/ocssw/glint_8c_source.html
        This corrects the sunglint contamination reflectances in the
        SeaWiFS 8 bands using the Cox & Munk model for the ocean
        sun glitter radiance distribution as function of the sea surface
        wind.
        Args:
            num_iter ():
            glint_coef ():
            airmass ():
            sza ():
            taur ():
            La ():
            num_iter, I, --- iteration number in the atmospheric corrections.
            nband, I, --- number of bands for the sensor (e.g., 8 for SeaWiFS).
            glint_coef, R, --- glitter radiance (F0=1) from Cox & Munk.
            air_mass, R, --- airmass value.
            mu0, R, --- cosine of the solar zenith angle.
            taur(nband), R, --- Rayleigh optical thicknesses.
            La(nband), R, --- aerosol reflectances at 8 SeaWiFS bands.
        Returns:
            TLg(nband), R, --- sunglint radiances at 8 SeaWiFS bands.
        """
        # 初始的气溶胶光学厚度有三种做法，一种是初始化为0.1，一种是使用多年的统计均值，一种是使用王梦华给的估算公式,
        # 默认使用王梦华的方式，根据865波段反射估算气溶胶
        if mode == 0:
            taua_ave = 0.1
        elif mode == 1:
            # 使用王梦华给的估算公式,
            rhoa = np.pi * La[:, :, -1] / F0[-1] / np.cos(np.deg2rad(sza))
            taua_ave = -0.383 * np.log(rhoa) - 1.6399
        elif mode == 2:
            try:
                taua_ave = taua[:, :, -1]
            except:
                taua_ave = taua

        # taua_ave = 0.1
        taua_min = 0.08
        rhoa_min = 0.01
        rhoa_min2 = 0.008
        rfac = 0.8
        glint_min = 0.0001
        nbands = F0.size
        TLg = np.zeros(shape=(sza.shape[0], sza.shape[1], nbands))
        iter_max = predefine.thresholds().glint_iter_max  # 从0次起算，1表示两次
        glint_coef[np.isnan(glint_coef)] = -1
        for i in range(nbands):
            TLg_lambda = np.zeros_like(sza)
            TLg_lambda[glint_coef <= glint_min] = 0.0
            TLg[:, :, i] = TLg_lambda
            del TLg_lambda
        del i
        refl_test = np.pi / np.cos(np.deg2rad(sza)) * (
                La[:, :, -1] / F0[-1] - glint_coef * np.exp(-(taur[:, :, -1] + taua_ave) * airmass))
        if iter_num < iter_max:
            taua_ave2 = np.empty_like(refl_test)
            refl_test[refl_test < 0.0001] = 0.0001
            judge = refl_test <= rhoa_min
            taua_ave2[judge] = -0.8 - 0.4 * np.log((10. * refl_test[judge]))
            try:
                taua_ave2[~judge] = taua_ave[~judge]
            except:
                taua_ave2[~judge] = taua_ave

        # 迭代次数决定tauc的计算方法：
        for i in range(nbands):
            if iter_num < iter_max:
                taua_c = taua_ave2
            else:
                # print(iter_num)
                taua_c = np.zeros_like(taua[:, :, -1])
                taua[:, :, -1][np.isnan(taua[:, :, -1])] = np.nanmean(taua[:, :, -1])
                judge = taua[:, :, -1] <= taua_min
                taua_c[judge] = -0.8 - 0.4 * np.log(taua[:, :, -1][judge])
                # taua_c[~judge] = taua[:, :, i][~judge]
                taua_c[~judge] = taua[:, :, -1][~judge]

            # 检查是否过校正
            if i == nbands - 1:
                refl_test = np.pi / np.cos(np.deg2rad(sza)) * (
                        La[:, :, -1] / F0[-1] - glint_coef * np.exp(-(taur[:, :, -1] + taua_c) * airmass))

            TLg_temp = np.zeros_like(refl_test)
            refl_test[np.isnan(refl_test)] = -999
            judge = (refl_test <= rhoa_min2) & (refl_test >= -100)
            TLg_temp[judge] = F0[i] * glint_coef[judge] * np.exp(-(taur[:, :, i][judge] + 1.5 * taua_c[judge]) * airmass[
                    judge])
            del judge
            judge = refl_test > rhoa_min2
            TLg_temp[judge] = F0[i] * glint_coef[judge] * np.exp(-(taur[:, :, i][judge] + taua_c[judge]) * airmass[
                    judge])
            del judge
            TLg_temp[TLg_temp < 0] = 0
            TLg[:, :, i] = TLg_temp
            del TLg_temp

        # /* Make sure there is no over-correction */ 这部分不参与迭代
        fac = TLg[:, :, -1] / La[:, :, -1]
        fac2 = TLg[:, :, -2] / La[:, :, -2]
        # print(np.mean(fac))
        # print(np.mean(fac2))
        # fac和fac2都可能为nan
        fac_0 = fac * 1.
        fac_0[np.isnan(fac_0)] = 999
        fac2_0 = fac2 * 1.
        fac2_0[np.isnan(fac2_0)] = -999
        # fac[fac < fac2] = fac2[fac < fac2]
        fac[fac_0 < fac2_0] = fac2[fac_0 < fac2_0]

        fac_00 = fac * 1.
        fac_00[np.isnan(fac_00)] = -999

        for i in range(nbands):
            # TLg[:, :, i][fac >= rfac] = rfac * TLg[:, :, i][fac >= rfac] / fac[fac >= rfac]
            TLg[:, :, i][fac_00 >= rfac] = rfac * TLg[:, :, i][fac_00 >= rfac] / fac[fac_00 >= rfac]
        return TLg


def main_exec(iter_num: int = None, vza=None, sza=None, vaa=None, saa=None, windspeed=None, winddirection=None,
              taur=None, La=None, F0=None, taua=None, mode: int = None):
    glintcoef_i, glintcoef_q, glintcoef_u = get_glint_coefficient.getglint_iqu(vza=vza, sza=sza, vaa=vaa, saa=saa,
                                                                               windspeed=windspeed,
                                                                               winddirection=winddirection)
    glintcoef_i[glintcoef_i > predefine.thresholds().glint_threshold] = np.nan
    # air_mass = airmass.ky_airmass(zenithangle=sza)
    air_mass = 1 / np.cos(np.deg2rad(sza)) + 1 / np.cos(np.deg2rad(vza))
    TLg = sunglint_contamination_reflectances.calcu(iter_num=iter_num, glint_coef=glintcoef_i, airmass=air_mass,
                                                    sza=sza, taur=taur, La=La, F0=F0.reshape(-1), taua=taua, mode=mode)
    return TLg
