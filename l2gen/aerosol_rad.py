# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7
@file: aerosol_radV2.py
@time: 2021/1/20 15:34
@desc: 气溶胶辐射计算函数
aerosol()气溶胶辐射计算的主函数

在气溶胶辐射的计算过程中，数据均为1行n列的数组
该文件参考了seadas源码的aerosol.c文件, 函数名和功能与aerosol.c文件相同
模型的选择方法与《Atmospheric Correction for Satellite Ocean Color Radiometry》描述的相似，与《Atmospheric correction of HJ-1 CCD imagery
over turbid lake waters》描述的相似。和Seadas的迭代选择方法有区别

插值函数可以预先在外部建立，
2023-10-1：最新版，优化了交叉定标的计算效率
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
# import sys
import airmass
# sys.path.append("..")
from sharepy import predefine, array_simplify


def aermod_interp_func(*args: str, target_value: str = None) -> dict:
    """
    先将插值函数计算出来
    """
    # 先将所有模型的插值函数计算出来
    function_set = {}
    for key in aerosol_models:
        # 插值函数
        aermod = aerosol_models[key]
        function_set[aermod['name']] = RegularGridInterpolator([aermod[idxi].reshape(-1) for idxi in args],
                                                               aermod[target_value],
                                                               bounds_error=False, method='linear', fill_value=None)
    return function_set


class Aerosol(object):
    def __init__(self, sza=None, vza=None, saa=None, vaa=None):
        self.sza = sza
        self.vza = vza
        self.saa = saa
        self.vaa = vaa
        self.sza_rad = np.deg2rad(self.sza)
        self.size = self.sza_rad.size
        self.vza_rad = np.deg2rad(self.vza)
        self.reaa = np.abs(self.saa - self.vaa)
        self.reaa[self.reaa > 180.] = self.reaa[self.reaa > 180.] - 180
        self.reaa_rad = np.deg2rad(self.reaa)

    def model_phase(self, aermod_index=None, band: float = None):
        """
        Parameters
        ----------
        aer_models : 气溶胶模型
        band :
        Returns
        -------
        根据模型计算的散射相函数
        """

        temp = np.sqrt((1. - np.cos(self.vza_rad) ** 2) * (1. - np.cos(self.sza_rad) ** 2)) * np.cos(self.reaa_rad)

        temp_1 = -np.cos(self.vza_rad) * np.cos(self.sza_rad) + temp
        temp_1[temp_1 < -1.] = -1.
        scatt1 = np.deg2rad(np.arccos(temp_1))
        temp_2 = np.cos(self.vza_rad) * np.cos(self.sza_rad) + temp
        temp_2[temp_2 > 1.] = 1.
        scatt2 = np.deg2rad(np.arccos(temp_2))

        # / *compute Fresnel coefficients * /
        nw = 1.334
        fres1 = self.fresnel_coef(angle=self.vza_rad, index=nw)
        fres2 = self.fresnel_coef(angle=self.sza_rad, index=nw)
        phase_funcs = aermod_interp_func('wave_lut', 'scatt_lut', target_value='phase_lut')
        bands = band.repeat(self.sza.size).reshape(1, -1)
        points1 = np.array([bands, scatt1]).T
        points1 = points1[:, 0, :]
        points2 = np.array([bands, scatt2]).T
        points2 = points2[:, 0, :]
        phase1 = np.zeros(shape=(1, self.sza.size))
        phase2 = np.zeros(shape=(1, self.sza.size))
        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            pts1 = points1[loc, :]
            pts2 = points2[loc, :]
            phase1[0, loc.reshape(-1)] = phase_funcs[index](pts1)
            phase2[0, loc.reshape(-1)] = phase_funcs[index](pts2)
        phase = np.exp(phase1.reshape(1, -1)) + np.exp(phase2.reshape(1, -1)) * (fres1 + fres2)
        return phase

    def fresnel_coef(self, angle=None, index=None):
        """
        Args:
            angle: 单位是弧度
            index:
        Returns:
        """
        mu = np.cos(angle)
        sq = np.sqrt(np.power(index, 2.0) - 1.0 + np.power(mu, 2.0))
        r2 = np.power((mu - sq) / (mu + sq), 2.0)
        q1 = (1.0 - np.power(mu, 2.0) - mu * sq) / (1.0 - np.power(mu, 2.0) + mu * sq)
        return r2 * (q1 * q1 + 1.0) / 2.0

    def model_epsilon(self, aermod_index=None, two_bands=np.array([750, 865])):
        """
        求任意一个波段与近红外波段的epsilon值
        Parameters
        ----------
        aer_models : 气溶胶模型
        bands :必须明确指定两个波段
        Returns
        -------

        """
        # print('model_epsilon')
        # rhoas1 = aertab->model[im]->albedo[iw] * phase[iw] * aertab->model[im]->extc[iw];
        # rhoas2 = aertab->model[im]->albedo[iwnir] * phase[iwnir] * aertab->model[im]->extc[iwnir];
        # aer_models = aer_models.reshape(1, -1)

        # 单次散射反照率albedo和消光系数extc
        albedo_funcs = aermod_interp_func('wave_lut', target_value='albedo_lut')
        extc_funcs = aermod_interp_func('wave_lut', target_value='extc_lut')
        bands2 = two_bands[1].repeat(self.sza.size).reshape(1, -1)
        points2 = np.array([bands2]).T
        points2 = points2[:, 0, :]

        bands1 = two_bands[0].repeat(self.sza.size).reshape(1, -1)
        points1 = np.array([bands1]).T
        points1 = points1[:, 0, :]

        albedo2 = np.zeros(shape=(1, self.sza.size))
        extc2 = np.zeros(shape=(1, self.sza.size))
        albedo1 = np.zeros(shape=(1, self.sza.size))
        extc1 = np.zeros(shape=(1, self.sza.size))
        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            pts2 = points2[loc, :]
            albedo2[0, loc.reshape(-1)] = albedo_funcs[index](pts2).reshape(-1)
            extc2[0, loc.reshape(-1)] = extc_funcs[index](pts2).reshape(-1)

            pts1 = points1[loc, :]
            albedo1[0, loc.reshape(-1)] = albedo_funcs[index](pts1).reshape(-1)
            extc1[0, loc.reshape(-1)] = extc_funcs[index](pts1).reshape(-1)

        # 1. 近红外长波段的理论模型rhoas
        albedo2 = albedo2.reshape(1, -1)
        extc2 = extc2.reshape(1, -1)
        phase2 = self.model_phase(aermod_index=aermod_index, band=two_bands[1])
        phase2 = phase2.reshape(1, -1)
        rhoas2 = phase2 * albedo2 * extc2

        # 2. 目标波段的理论模型rhoas
        phase1 = self.model_phase(aermod_index=aermod_index, band=two_bands[0])
        phase1 = phase1.reshape(1, -1)
        albedo1 = albedo1.reshape(1, -1)
        extc1 = extc1.reshape(1, -1)
        rhoas1 = phase1 * albedo1 * extc1

        epsilon = rhoas1 / rhoas2  # xxx/865

        return epsilon

    def rh_select_models(self, relative_humidity=None):
        """
        根据湿度选择20个模型
        Parameters
        ----------
        relative_humidity :
        aerosol_models :

        Returns
        -------

        """
        # print('rh_select_models')
        # 假设有n个像元，则根据湿度对每个像元需要找到20个模型

        # 湿度边界
        relative_humidity_unit = np.array([[30], [50], [70], [75], [80], [85], [90], [95]])
        relative_humidity_lut = np.tile(relative_humidity_unit, (1, relative_humidity.shape[1]))
        diff = relative_humidity_lut - relative_humidity

        # 寻找上边界，现将小于0的转为999，然后找最小值
        diff1 = diff * 1.
        diff1[diff1 < 0] = 999
        t1 = diff1.argmin(axis=0)
        # 初选模型时，每个湿度条件下的10种模型都要选择。因此t*10，t*10+1，...t*10+9，就是选择的10各模型的位置
        rh_up_start = t1.reshape(1, -1) * 10  # rh_up_start表示一个位置
        diff2 = diff * 1.
        diff2[diff2 > 0] = -999
        t2 = diff1.argmax(axis=0)
        rh_low_start = t2.reshape(1, -1) * 10
        return rh_up_start, rh_low_start

    def model_taua(self, aermod_index=None, rhoas_nir2=None, nir_l_wave: float = None):
        """
        Parameters
        ----------
        aer_models :
        rhoas_nir2 :
        bands :

        Returns
        -------
            计算所有波段的气溶胶光学厚度，Gordon指出，气溶胶光学厚度与气溶胶模型相关，与单次散射反照率、散射相函数相关
        """
        # print('model_taua')
        phase = self.model_phase(aermod_index=aermod_index, band=nir_l_wave)
        albedo_funcs = aermod_interp_func('wave_lut', target_value='albedo_lut')
        bands = nir_l_wave.repeat(self.sza_rad.size).reshape(1, -1)
        points = np.array([bands]).T
        points = points[:, 0, :]
        albedo = np.zeros(shape=(1, self.sza_rad.size))
        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            pts = points[loc, :]
            # 单次散射反照率albedo
            albedo[0, loc.reshape(-1)] = albedo_funcs[index](pts).reshape(-1)
        albedo = albedo.reshape(1, -1)

        # /* get aerosol optical thickness at longest sensor wavelength */
        aot_nir2 = rhoas_nir2 * 4. * np.cos(self.sza_rad) * np.cos(self.vza_rad) / albedo / phase

        # 模型的气溶胶光学厚度
        # /* get aerosol optical thickness at all other table wavelengths */
        # 从模型分别求消aot
        # 读取消光系数
        models_set = {}
        for key in aerosol_models:
            # 插值函数
            aermod = aerosol_models[key]
            models_set[aermod['name']] = aermod
        extc = np.array([models_set[int(index)]['extc_lut'].reshape(-1) for index in aermod_index[0, :]])
        extc = extc.T
        # 其它波段的气溶胶光学厚度
        aot = aot_nir2 * extc[:, :] / extc[-1, :, ]
        return aot

    def rhoa_to_rhoas(self, aermod_index: np.ndarray = None, rho_a=None, band=None, aeroob=1):
        """
        目前用于长波近红外的气溶胶辐射多次转单次
        Parameters
        ----------
        aeroob :
        aermod_index :模型指数（模型位置）
        rho_a :气溶胶多次散射
        band : 单个波段组成的数组，np.array([869])

        Returns
        -------
        气溶胶单次散射
        """
        # print('rhoa_to_rhoas')
        # aer_models = aer_models.reshape(1, -1)

        acost_funcs = aermod_interp_func('wave_lut', 'solz_lut', 'phi_lut', 'senz_lut', target_value='acost_lut')
        bcost_funcs = aermod_interp_func('wave_lut', 'solz_lut', 'phi_lut', 'senz_lut', target_value='bcost_lut')
        ccost_funcs = aermod_interp_func('wave_lut', 'solz_lut', 'phi_lut', 'senz_lut', target_value='ccost_lut')
        bands = band.repeat(self.sza_rad.size).reshape(1, -1)
        points = np.array([bands, self.sza, self.reaa, self.vza]).T
        points = points[:, 0, :]
        # aermod_index=np.array([26, 26, 13, 18, 15, 55, 45, 13, 18, 55])
        # aermod_index=aermod_index.reshape(1,-1)
        a = np.zeros(shape=(1, self.sza_rad.size))
        b = np.zeros(shape=(1, self.sza_rad.size))
        c = np.zeros(shape=(1, self.sza_rad.size))
        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            pts = points[loc, :]
            a[0, loc.reshape(-1)] = acost_funcs[index](pts)
            b[0, loc.reshape(-1)] = bcost_funcs[index](pts)
            c[0, loc.reshape(-1)] = ccost_funcs[index](pts)

        # ==========================多次转单次================================================
        ss_part1 = np.ones(shape=(1, rho_a.shape[1])) * (-1.)
        ss_part2 = np.ones(shape=(1, rho_a.shape[1])) * (-1.)
        ss_part3 = np.ones(shape=(1, rho_a.shape[1])) * (-1.)
        judge = (rho_a < 1.e-20) & (rho_a > 0)
        ss_part1[judge] = rho_a[judge]
        del judge
        # 不属于这部分的全部掩膜掉nan
        ss_part1[(ss_part1 > 1.e-20) | (ss_part1 < 0.)] = np.nan

        f1 = b ** 2 - 4 * c * (a - np.log(rho_a))
        ss_part2[~np.isnan(ss_part1)] = np.nan
        ss_part2[f1 < 1.e-5] = np.nan
        judge = (f1 > 1.e-5) & (np.abs(c) > 1.e-20) & (~np.isnan(ss_part2))
        ss_part2[judge] = np.exp(0.5 * (-b[judge] + np.sqrt(f1[judge])) / c[judge])
        # ss_part2[(f1 < 1.e-5)&(abs(c) > 1.e-20)] = np.nan
        # 现将nan赋值为-1，然后将所有小于0 的数赋值为nan。这一方法虽然重复了步骤，但是可以避免nan警告，因为nan是不能参与比较的
        ss_part2[np.isnan(ss_part2)] = -1
        ss_part2[ss_part2 < 0] = np.nan

        ss_part3[~np.isnan(ss_part1)] = np.nan
        ss_part3[~np.isnan(ss_part2)] = np.nan
        ss_part3[f1 < 1.e-5] = np.nan
        judge = (f1 > 1.e-5) & (np.abs(a) > 1.e-20) & (np.abs(b) > 1.e-20) & (~np.isnan(ss_part3))
        try:
            ss_part3[judge] = np.power((rho_a[judge])(a[judge]), 1 / b[judge])
            del judge
        except:
            pass
        ss_part3[np.isnan(ss_part3)] = -1
        ss_part3[ss_part3 < 0] = np.nan
        # 三种情况下的多次转单次确定了
        ss_aero = np.nanmean(np.dstack((ss_part1, ss_part2, ss_part3)), 2)
        # ===============================================================================================
        # 将余下的空值补上多次散射的值作为单次散射
        ss_aero[np.isnan(ss_aero)] = -1
        ss_aero[ss_aero <= 0] = np.nan
        ss_aero[np.isnan(ss_aero)] = rho_a[np.isnan(ss_aero)]
        return ss_aero / aeroob

    def rhoas_to_rhoa(self, aermod_index: np.ndarray = None, rhoas=None, band=None, aeroob=1):
        """

        Parameters
        ----------
        aer_models :
        rhoas :
        band :
        aeroob : /* aeroob - out-of-band water-vapor correction   暂时不做
        Returns
        -------
        """

        # print('rhoas_to_rhoa')
        # aer_models = aer_models.reshape(1, -1)

        acost_funcs = aermod_interp_func('wave_lut', 'solz_lut', 'phi_lut', 'senz_lut', target_value='acost_lut')
        bcost_funcs = aermod_interp_func('wave_lut', 'solz_lut', 'phi_lut', 'senz_lut', target_value='bcost_lut')
        ccost_funcs = aermod_interp_func('wave_lut', 'solz_lut', 'phi_lut', 'senz_lut', target_value='ccost_lut')
        bands = band.repeat(self.sza_rad.size).reshape(1, -1)
        points = np.array([bands, self.sza, self.reaa, self.vza]).T
        points = points[:, 0, :]
        # aermod_index=np.array([26, 26, 13, 18, 15, 55, 45, 13, 18, 55])
        # aermod_index=aermod_index.reshape(1,-1)
        a = np.zeros(shape=(1, self.sza_rad.size))
        b = np.zeros(shape=(1, self.sza_rad.size))
        c = np.zeros(shape=(1, self.sza_rad.size))
        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            pts = points[loc, :]
            a[0, loc.reshape(-1)] = acost_funcs[index](pts)
            b[0, loc.reshape(-1)] = bcost_funcs[index](pts)
            c[0, loc.reshape(-1)] = ccost_funcs[index](pts)

        # lnrhoas = log(rhoas[iw] * aeroob(sensorID, iw, geom->airmass[ig], cf, wv));
        # rhoa[iw] = exp(a + b * lnrhoas + c * lnrhoas * lnrhoas);
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        c = c.reshape(1, -1)
        lnrhoas = np.log(rhoas * aeroob)
        rhoa = np.exp(a + b * lnrhoas + c * lnrhoas * lnrhoas)
        rhoa[rhoas < 1e-20] = rhoas[rhoas < 1e-20]
        return rhoa

    def model_transmittance(self, aermod_index: np.ndarray = None, bands=None, aot=None):
        """
        bands: 所有波长
        aot: 所有波段的aot
        aot和bands应该保持相同的
        两个经验系数a/b：t∗(θv) = A(θv) exp[−B(θv)τa]
        """
        # print('model_transmittance()')
        # rh_models = rh_models.reshape(1, -1)

        dtran_a0_funcs = aermod_interp_func('wave_lut', 'dtran_theta_lut', target_value='dtran_a0_lut')
        dtran_b0_funcs = aermod_interp_func('wave_lut', 'dtran_theta_lut', target_value='dtran_b0_lut')
        dtran_a_funcs = aermod_interp_func('wave_lut', 'dtran_theta_lut', target_value='dtran_a_lut')
        dtran_b_funcs = aermod_interp_func('wave_lut', 'dtran_theta_lut', target_value='dtran_b_lut')

        a0 = np.zeros(shape=(bands.size, self.size))
        b0 = np.zeros(shape=(bands.size, self.size))
        a = np.zeros(shape=(bands.size, self.size))
        b = np.zeros(shape=(bands.size, self.size))
        points1 = np.zeros(shape=(self.size, 2))
        points1[:, 1] = self.sza
        points2 = np.zeros(shape=(self.size, 2))
        points2[:, 1] = self.vza

        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            for j, band in enumerate(bands):
                points1[:, 0] = band
                points2[:, 0] = band
                pts1 = points1[loc, :]
                pts2 = points2[loc, :]
                a0[j, loc.reshape(-1)] = dtran_a0_funcs[index](pts1)
                b0[j, loc.reshape(-1)] = dtran_b0_funcs[index](pts1)
                a[j, loc.reshape(-1)] = dtran_a_funcs[index](pts2)
                b[j, loc.reshape(-1)] = dtran_b_funcs[index](pts2)

        # 漫射透过率
        t_solar = a0 * np.exp(-b0 * aot)
        t_solar[t_solar > 1] = 1
        t_solar[t_solar < 1e-5] = 1e-5
        t_sensor = a * np.exp(-b * aot)
        t_sensor[t_sensor > 1] = 1
        t_sensor[t_sensor < 1e-5] = 1e-5

        return t_sensor, t_solar

    def diff_tran(self, aer_model_min=None, aer_model_max=None, delta=None, tauamin=None, tauamax=None, bands=None,
                  taur=None, pressure=None):
        """
        :param aer_model_min:
        :param aer_model_max:
        :param delta:
        :param pressure:
        :param tauamin:
        :param tauamax:
        :param bands:
        :param taur:
        :return:
        """

        # print('diff_tran()')
        t_sensor_low, t_solar_low = self.model_transmittance(aermod_index=aer_model_min, bands=bands, aot=tauamin)
        t_sensor_up, t_solar_up = self.model_transmittance(aermod_index=aer_model_max, bands=bands, aot=tauamax)

        taua = tauamin * (1.0 - delta) + tauamax * delta
        t_sensor = t_sensor_low * (1.0 - delta) + t_sensor_up * delta
        t_solar = t_solar_low * (1.0 - delta) + t_solar_up * delta

        # 气压校正
        taur = taur.reshape(-1, 1)
        t_sensor = t_sensor * np.exp(-0.5 * taur / np.cos(self.sza_rad) * (pressure / 1013.25 - 1))
        t_solar = t_solar * np.exp(-0.5 * taur / np.cos(self.vza_rad) * (pressure / 1013.25 - 1))

        # 大气路径校正
        airmass_sph = airmass.ky_airmass(self.sza) + airmass.ky_airmass(self.vza)
        airmass_plp = airmass.pp_airmass(self.sza) + airmass.pp_airmass(self.vza)
        t_sensor = np.power(t_sensor, airmass_sph / airmass_plp)
        t_solar = np.power(t_solar, airmass_sph / airmass_plp)

        return taua, t_sensor, t_solar

    def wangaer(self, bands=None, aer_model_min=None, aer_model_max=None, rhoa_nir2=None, delta=None, nirl_num=None):
        """
        wangaer() - compute aerosol reflectance using Gordon & Wang 1994 algorithm
        Returns
        -------
        """
        # rh_models_low = rh_models_low.reshape(1, -1)
        # rh_models_up = rh_models_up.reshape(1, -1)
        # print('wangaer')
        # /* get SS aerosol reflectance at longest wavelength for the two models */
        rhoasmin_nir2 = self.rhoa_to_rhoas(aermod_index=aer_model_min, rho_a=rhoa_nir2, band=bands[nirl_num], aeroob=1)
        rhoasmax_nir2 = self.rhoa_to_rhoas(aermod_index=aer_model_max, rho_a=rhoa_nir2, band=bands[nirl_num], aeroob=1)

        # 1 根据确定的模型计算各各波段的epsilon->气溶胶单次散射->气溶胶多次散射
        rhoa = np.full(shape=(bands.size, self.size), fill_value=np.nan)
        for num, band in enumerate(bands):
            epsmax = self.model_epsilon(aermod_index=aer_model_max, two_bands=np.array([band, bands[nirl_num]]))
            epsmin = self.model_epsilon(aermod_index=aer_model_min, two_bands=np.array([band, bands[nirl_num]]))

            # /* compute SS aerosol reflectance in all bands */
            # rhoasmax = rhoasmax_nir2 * epsmax
            # rhoasmin = rhoasmin_nir2 * epsmin
            rhoasmax = rhoasmax_nir2 * epsmax
            rhoasmin = rhoasmin_nir2 * epsmin

            # /* compute MS aerosol reflectance in visible bands */
            rhoamax = self.rhoas_to_rhoa(aermod_index=aer_model_max, rhoas=rhoasmax, band=np.array([band]), aeroob=1)
            rhoamin = self.rhoas_to_rhoa(aermod_index=aer_model_min, rhoas=rhoasmin, band=np.array([band]), aeroob=1)
            # /* interpolate between upper and lower-bounding models */
            rhoa[num, :] = rhoamin * (1.0 - delta) + rhoamax * delta

        return rhoa, rhoasmin_nir2, rhoasmax_nir2

    def model_select_mobley(self, rho_a_nir1=None, rho_a_nir2=None, relative_humidity=None, aerosol_models=None,
                            bands=np.array([750, 865])):
        """
        这个方法与《Atmospheric Correction for Satellite Ocean Color Radiometry》描述的相似，所以命名为model_select_mobley
        mobley在该书第9章（53页）的表述认为epsilon(lambda1,lambda2)可直接等于这两个波段气溶胶多次散射的比值。这与wang的模型选择方法是有区别的
        Parameters
        ----------
        rho_a_nir1 : 第一个近红外波段的气溶胶辐射
        rho_a_nir2 :第二个近红外波段的气溶胶辐射
        relative_humidity :相对湿度
        aerosol_models :80个气溶胶模型
        bands :两个近红外波段的波长（单位与气溶胶模型里面的波长wave的相同，一般为nm）
        Returns
        气溶胶模型和delta（两个模型占比）
        -------
        """
        # 第一步. 根据湿度选出预选模型
        '''
            查找表各参数
        '''
        # print('model_select_mobley')
        rh_models = self.rh_select_models(relative_humidity=relative_humidity, aerosol_models=aerosol_models)

        # 第二步：计算两个近红外波段的模型epsilon
        # 为了计算理解和函数通用，逐行计算，因此这里要循环20次
        epsilon_model_rh = np.zeros(shape=(20, self.size))
        for model_i in range(20):
            epsilon_model_rh[model_i, :] = self.model_epsilon(aer_models=rh_models[model_i, :], two_bands=bands)
        epsilon_measure = rho_a_nir1 / rho_a_nir2
        # 找到上下边界值
        # 如果边界溢出，需要找到最接近的,在这里用c_fill去填充,另外一种情况是一列均为nan
        # 后续应该把边界溢出的像元贴上标签，这里暂时未处理
        # idx_rh_epsilonlow：模型的序号，表示潜在的模型位于20个模型中第几个位置，这里为下边界模型
        c = epsilon_model_rh - epsilon_measure
        c[c > 0] = -999
        c[np.isnan(c)] = -999
        try:
            idx_rh_epsilonlow = np.nanargmax(c, axis=0).reshape(1, -1)
        except:
            idx_rh_epsilonlow_arr = np.array(
                [[0, 0] if np.all(np.isnan(c[:, i])) else [1, np.nanargmax(c[:, i])] for i in
                 range(c.shape[1])]).T
            # [0,0]无效气溶胶校正标记，为了执行，选择的气溶胶模型为第一个；[1,np.nanargmax(c[:, i])]为有效校正
            idx_rh_epsilonlow = idx_rh_epsilonlow_arr[1, :]
            mark_invalid_1 = idx_rh_epsilonlow_arr[0, :]
        del c

        c = epsilon_model_rh - epsilon_measure
        c[c < 0] = 999
        c[np.isnan(c)] = 999
        try:
            idx_rh_epsilonup = np.nanargmin(c, axis=0).reshape(1, -1)
        except:
            idx_rh_epsilonup_arr = np.array(
                [[0, 0] if np.all(np.isnan(c[:, i])) else [1, np.nanargmin(c[:, i])] for i in
                 range(c.shape[1])]).T
            idx_rh_epsilonup = idx_rh_epsilonup_arr[1, :]
            mark_invalid_2 = idx_rh_epsilonup_arr[0, :]
        del c

        # 如果epsilonup或者epsilonlow均大于/小于epsilon_measure,则选择一个最近的值作为epsilonup/low
        c = np.abs(epsilon_model_rh - epsilon_measure)
        idx_rh_epsilon_alternate = np.nanargmin(c, axis=0).reshape(1, -1)
        del c

        # 取出上下边界模型的epsilon值
        epsilonup = epsilon_model_rh[list(idx_rh_epsilonup), list(range(epsilon_measure.shape[1]))]
        epsilonup = epsilonup.reshape(1, -1)
        epsilonlow = epsilon_model_rh[list(idx_rh_epsilonlow), list(range(epsilon_measure.shape[1]))]
        epsilonlow = epsilonlow.reshape(1, -1)

        # 如果epsilonlow和epsilonup同时大于（或者小于）epsilon_measure，则设置选择的模型只有一个，即最靠近epsilon_measure的那个，
        # 如果靠近的是epsilonlow，则 delta此时等于0，如果靠近的是epsilonup，则 delta此时等于1
        delta = np.zeros_like(epsilonup)
        judge = (epsilonlow == epsilonup)
        delta[~judge] = (epsilon_measure[~judge] - epsilonlow[~judge]) / (epsilonup[~judge] - epsilonlow[~judge])
        delta[(epsilonlow > epsilon_measure) & (epsilonup > epsilon_measure)] = 0
        delta[(epsilonlow < epsilon_measure) & (epsilonup < epsilon_measure)] = 1
        delta[epsilonlow == epsilonup] = 0
        # delta = (epsilon_measure - epsilonlow) / (epsilonup - epsilonlow)
        delta = np.nan_to_num(delta, nan=0, posinf=0, neginf=0)
        del judge

        # 将气溶胶模型的标号也需要修改，以保证选择到正确的气溶胶模型
        judge = (epsilonlow > epsilon_measure) & (epsilonup > epsilon_measure)
        idx_rh_epsilonup[judge] = idx_rh_epsilon_alternate[judge]
        judge = (epsilonlow < epsilon_measure) & (epsilonup < epsilon_measure)
        idx_rh_epsilonup[judge] = idx_rh_epsilon_alternate[judge]
        judge = (epsilonlow > epsilon_measure) & (epsilonup > epsilon_measure)
        idx_rh_epsilonlow[judge] = idx_rh_epsilon_alternate[judge]
        judge = (epsilonlow < epsilon_measure) & (epsilonup < epsilon_measure)
        idx_rh_epsilonlow[judge] = idx_rh_epsilon_alternate[judge]

        # 修正取出上下边界模型的epsilon值
        epsilonup = epsilon_model_rh[list(idx_rh_epsilonup), list(range(epsilon_measure.shape[1]))]
        epsilonup = epsilonup.reshape(1, -1)
        epsilonlow = epsilon_model_rh[list(idx_rh_epsilonlow), list(range(epsilon_measure.shape[1]))]
        epsilonlow = epsilonlow.reshape(1, -1)

        # 处理非法像元标记
        try:
            mark_invalid = mark_invalid_1 * mark_invalid_2
        except:
            mark_invalid = np.ones_like(delta)

        # # 计算出865nm的气溶胶单次散射
        # ss_nir2 = (1 - delta) * np.choose(idx_rh_epsilonlow, ss_nir2_rh) + delta * np.choose(idx_rh_epsilonup,
        #                                                                                      ss_nir2_rh)
        # np.choose会自动匹配行列
        rh_models_low = np.choose(idx_rh_epsilonlow, rh_models)
        rh_models_up = np.choose(idx_rh_epsilonup, rh_models)

        aer_model_min = np.full_like(delta, fill_value=-32767)
        aer_model_max = np.full_like(delta, fill_value=-32767)
        for i in range(delta[0, :]):
            aer_model_min[0, i] = rh_models_low[0, i].get('name')
            aer_model_max[0, i] = rh_models_up[0, i].get('name')

        return aer_model_min, aer_model_max, rh_models_low, rh_models_up, epsilonlow, epsilonup, delta, mark_invalid

    def model_select_angstrom(self, angstrom_measure: float = None, relative_humidity=None, band520: int = 520):
        """
        :param angstrom: 实测的angstrom
        :param aermod_index: 预选的气溶胶模型，可以根据湿度预选20个，或者全部的80个。[0-79]shape=(20 or 80,angstrom.size)
        :return:
        """
        # 根据相对湿度选择出20个模型
        rh_up_start, rh_low_start = self.rh_select_models(relative_humidity=relative_humidity)
        angstrom_model_rh = np.zeros(shape=(20, angstrom_measure.shape[1]))
        for i in range(20):
            if i < 10:
                aermod_index = rh_up_start + i
            else:
                aermod_index = rh_low_start + i - 10
            angstrom_model_rh[i, :] = self.model_angstrom(aermod_index=aermod_index, band=band520)

        # 选择上下两个边界模型
        diff = angstrom_measure - angstrom_model_rh
        diff[np.isnan(diff)] = 999
        diff[diff < 0] = 999
        idx_bracket_low = np.nanargmin(diff, axis=0).reshape(1, -1)
        del diff
        diff = angstrom_measure - angstrom_model_rh
        diff[np.isnan(diff)] = -999
        diff[diff > 0] = -999
        idx_bracket_up = np.nanargmax(diff, axis=0).reshape(1, -1)
        del diff
        ang_mod_low = np.choose(idx_bracket_low, angstrom_model_rh)
        ang_mod_up = np.choose(idx_bracket_up, angstrom_model_rh)

        # 如果epsilonlow和epsilonup同时大于（或者小于）epsilon_measure，则设置选择的模型只有一个，即最靠近epsilon_measure的那个，
        # 如果靠近的是epsilonlow，则 delta此时等于0，如果靠近的是epsilonup，则 delta此时等于1
        ang_mod_up = ang_mod_up.reshape(1, -1)
        ang_mod_low = ang_mod_low.reshape(1, -1)
        ang_mod_up[np.isnan(ang_mod_up)] = 999
        ang_mod_low[np.isnan(ang_mod_low)] = 999

        delta = np.zeros_like(ang_mod_up)
        judge = (ang_mod_low == ang_mod_up)
        delta[~judge] = (angstrom_measure[~judge] - ang_mod_low[~judge]) / (ang_mod_up[~judge] - ang_mod_low[~judge])
        delta = np.nan_to_num(delta, nan=0, posinf=0, neginf=0)
        delta[(ang_mod_low > angstrom_measure) & (ang_mod_up > angstrom_measure)] = 1
        delta[(ang_mod_low < angstrom_measure) & (ang_mod_up < angstrom_measure)] = 0
        delta[ang_mod_low == ang_mod_up] = 0

        # 最终的模型指数
        aer_model_min = np.zeros(shape=(1, angstrom_measure.size))
        aer_model_min[idx_bracket_low < 10] = idx_bracket_low[idx_bracket_low < 10] + rh_up_start[idx_bracket_low < 10]
        aer_model_min[idx_bracket_low >= 10] = idx_bracket_low[idx_bracket_low >= 10] + rh_low_start[
            idx_bracket_low >= 10] - 10

        aer_model_max = np.zeros(shape=(1, angstrom_measure.size))
        aer_model_max[idx_bracket_up < 10] = idx_bracket_up[idx_bracket_up < 10] + rh_up_start[idx_bracket_up < 10]
        aer_model_max[idx_bracket_up >= 10] = idx_bracket_up[idx_bracket_up >= 10] + rh_low_start[
            idx_bracket_up >= 10] - 10
        return aer_model_min, aer_model_max, ang_mod_low, ang_mod_up, delta

    def model_select_wang(self, relative_humidity=None, rho_a_nir1=None, rho_a_nir2=None,
                          nir_bands=np.array([750, 865])):
        """
        这这个方法的描述见《珠江口二类水体MODIS数据大气校正》
        该方法反演每种气溶胶单次散射，然后计算epsilon。再通过模型的单次散射反照率计算epsilon。比较两个epsilon
        """
        rh_up_start, rh_low_start = self.rh_select_models(relative_humidity=relative_humidity)
        eps_retri = np.zeros(shape=(20, self.size))
        epsilon_model_rh = np.zeros(shape=(20, self.size))
        for i in range(20):
            if i < 10:
                aermod_index = rh_up_start + i
            else:
                aermod_index = rh_low_start + i - 10
            rhoas_nirs_retri = self.rhoa_to_rhoas(aermod_index=aermod_index, rho_a=rho_a_nir1, band=nir_bands[0],
                                                  aeroob=1)
            rhoas_nirl_retri = self.rhoa_to_rhoas(aermod_index=aermod_index, rho_a=rho_a_nir2, band=nir_bands[1],
                                                  aeroob=1)
            eps_retri_temp = rhoas_nirs_retri / rhoas_nirl_retri
            eps_retri[i, :] = eps_retri_temp
            epsilon_model_rh[i, :] = self.model_epsilon(aermod_index=aermod_index, two_bands=nir_bands)

        eps_ave = np.nanmean(eps_retri, axis=0).reshape(1, -1)
        column_idx = np.arange(0, self.size)
        for i in range(8):
            diff = np.abs(eps_ave - epsilon_model_rh)
            # 去掉两个最大的
            idx1 = np.nanargmax(diff, axis=0)
            diff[idx1, column_idx] = np.nan
            eps_retri[idx1, column_idx] = np.nan
            epsilon_model_rh[idx1, column_idx] = np.nan
            idx2 = np.nanargmax(diff, axis=0)
            diff[idx2, column_idx] = np.nan
            eps_retri[idx2, column_idx] = np.nan
            epsilon_model_rh[idx2, column_idx] = np.nan
            # 重新确定均值
            tot_err = np.nansum(np.abs(diff), axis=0).reshape(1, -1)
            wt = 1 - np.abs(diff) / tot_err
            eps_ave = np.nansum(eps_retri * wt, axis=0).reshape(1, -1) / (20 - (i + 1) * 2 - 1)

        # 剩下四个模型，选择上下两个边界模型
        diff = eps_ave - epsilon_model_rh
        diff[np.isnan(diff)] = 999
        diff[diff < 0] = 999
        # try:
        idx_bracket_low = np.nanargmin(diff, axis=0).reshape(1, -1)
        # except:
        #     idx_rh_epsilonlow_arr = np.array(
        #         [[0, 0] if np.all(np.isnan(diff[:, i])) else [1, np.nanargmax(diff[:, i])] for i in
        #          range(diff.shape[1])]).T
        #     # [0,0]无效气溶胶校正标记，为了执行，选择的气溶胶模型为第一个；[1,np.nanargmax(c[:, i])]为有效校正
        #     idx_bracket_low= idx_rh_epsilonlow_arr[1, :]
        #     mark_invalid_1 = idx_rh_epsilonlow_arr[0, :]

        del diff
        diff = eps_ave - epsilon_model_rh
        diff[np.isnan(diff)] = -999
        diff[diff > 0] = -999

        # try:
        idx_bracket_up = np.nanargmax(diff, axis=0).reshape(1, -1)
        # except:
        #     idx_rh_epsilonup_arr = np.array(
        #         [[0, 0] if np.all(np.isnan(diff[:, i])) else [1, np.nanargmin(diff[:, i])] for i in
        #          range(diff.shape[1])]).T
        #     # [0,0]无效气溶胶校正标记，为了执行，选择的气溶胶模型为第一个；[1,np.nanargmax(c[:, i])]为有效校正
        #     idx_bracket_up = idx_rh_epsilonup_arr[1, :]
        #     mark_invalid_1 = idx_rh_epsilonup_arr[0, :]
        # del diff

        eps_mod_low = np.choose(idx_bracket_low, epsilon_model_rh)
        eps_mod_up = np.choose(idx_bracket_up, epsilon_model_rh)

        # 如果epsilonlow和epsilonup同时大于（或者小于）epsilon_measure，则设置选择的模型只有一个，即最靠近epsilon_measure的那个，
        # 如果靠近的是epsilonlow，则 delta此时等于0，如果靠近的是epsilonup，则 delta此时等于1
        eps_mod_up = eps_mod_up.reshape(1, -1)
        eps_mod_low = eps_mod_low.reshape(1, -1)
        eps_mod_up[np.isnan(eps_mod_up)] = 999
        eps_mod_low[np.isnan(eps_mod_low)] = 999

        delta = np.zeros_like(eps_mod_up)
        judge = ~(eps_mod_low == eps_mod_up)
        delta[judge] = (eps_ave[judge] - eps_mod_low[judge]) / (eps_mod_up[judge] - eps_mod_low[judge])
        delta = np.nan_to_num(delta, nan=0, posinf=0, neginf=0)
        delta[(eps_mod_low > eps_ave) & (eps_mod_up > eps_ave)] = 1

        delta[(eps_mod_low < eps_ave) & (eps_mod_up < eps_ave)] = 0
        delta[eps_mod_low == eps_mod_up] = 0

        # 最终的模型指数
        aer_model_min = np.zeros(shape=(1, self.size))
        judge = idx_bracket_low < 10
        aer_model_min[judge] = idx_bracket_low[judge] + rh_up_start[judge]
        del judge
        judge = idx_bracket_low >= 10
        aer_model_min[judge] = idx_bracket_low[judge] + rh_low_start[judge] - 10
        del judge
        aer_model_max = np.zeros(shape=(1, self.size))
        judge = idx_bracket_up < 10
        aer_model_max[judge] = idx_bracket_up[judge] + rh_up_start[judge]
        del judge
        judge = idx_bracket_up >= 10
        aer_model_max[judge] = idx_bracket_up[judge] + rh_low_start[judge] - 10

        return aer_model_min, aer_model_max, eps_mod_low, eps_mod_up, delta

    def model_angstrom(self, aermod_index: np.ndarray = None, band=None):
        """
        :param aermod_index:
        :param band:
        :return:
        """
        angstrom_funcs = aermod_interp_func('wave_lut', target_value="angstrom_lut")
        bands = band.repeat(aermod_index.size).reshape(1, -1)
        points = np.array([bands]).T
        points = points[:, 0, :]
        angstrom_model = np.zeros(shape=(1, aermod_index.size))
        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            pts = points[loc, :]
            angstrom_model[0, loc.reshape(-1)] = angstrom_funcs[index](pts).reshape(-1)
        return angstrom_model

    def model_albedo(self, aermod_index: np.ndarray = None, band=None):
        """
        :param aermod_index:
        :param band:
        :return:
        """
        albedo_funcs = aermod_interp_func('wave_lut', target_value='albedo_lut')
        bands = band.repeat(aermod_index.size).reshape(1, -1)
        points = np.array([bands]).T
        points = points[:, 0, :]
        albedo_model = np.zeros(shape=(1, aermod_index.size))
        for index in np.unique(aermod_index):
            # 把同一个模型的反演出来
            loc = np.argwhere(aermod_index == index)[:, 1]
            pts = points[loc, :]
            albedo_model[0, loc.reshape(-1)] = albedo_funcs[index](pts).reshape(-1)
        return albedo_model

    def wang_method(self, rho_a_nir1=None, rho_a_nir2=None, relative_humidity=None,
                    bands=None, nirs_num: int = None, nirl_num: int = None, taur=None, pressure=None):
        # global aerosol_models
        # aerosol_models = aerosol_models_info

        # 2 选择气溶胶模型
        # print('====selecting aerosol models ...')
        aer_model_min, aer_model_max, epsilonlow, epsilonup, delta = \
            self.model_select_wang(relative_humidity=relative_humidity, rho_a_nir1=rho_a_nir1, rho_a_nir2=rho_a_nir2,
                                   nir_bands=np.array([bands[nirs_num], bands[nirl_num]]))

        # print('====computing aerosol reflectance ...')
        # 计算各波段的气溶胶辐射
        rhoa, rhoasmin_nir2, rhoasmax_nir2 = self.wangaer(delta=delta, bands=bands, nirl_num=nirl_num,
                                                          aer_model_min=aer_model_min, aer_model_max=aer_model_max,
                                                          rhoa_nir2=rho_a_nir2)
        # print('====computing taua/t_sen/tsol ...')
        # 计算出模型里所有波长的taua，在这里波段给一个nir_l
        taua_max = self.model_taua(aermod_index=aer_model_max, rhoas_nir2=rhoasmax_nir2, nir_l_wave=bands[nirl_num])
        taua_min = self.model_taua(aermod_index=aer_model_min, rhoas_nir2=rhoasmin_nir2, nir_l_wave=bands[nirl_num])

        taua, t_sensor, t_solar = self.diff_tran(aer_model_min=aer_model_min, aer_model_max=aer_model_max,
                                                 pressure=pressure, delta=delta, bands=bands, tauamin=taua_min,
                                                 tauamax=taua_max, taur=taur)
        return rhoa, taua, t_sensor, t_solar, aer_model_min, aer_model_max, delta, rhoasmin_nir2, rhoasmax_nir2


def cross_calibration(delta=None, rhoa_nirl=None, sza=None, vza=None, saa=None, vaa=None, aer_model_max=None,
                      aer_model_min=None, aerosol_models_info=None, F0=None, pressure=None, taur=None,
                      bands=None, nirl_num=None, sza_ref=None, vza_ref=None, saa_ref=None, vaa_ref=None):
    """
    使用气溶胶模型
    """
    arr = np.full(shape=(delta.shape[0], delta.shape[1], 13), fill_value=np.nan)
    arr[:, :, 0] = delta
    arr[:, :, 1] = rhoa_nirl
    arr[:, :, 2] = sza
    arr[:, :, 3] = vza
    arr[:, :, 4] = saa
    arr[:, :, 5] = vaa
    arr[:, :, 6] = sza_ref
    arr[:, :, 7] = vza_ref
    arr[:, :, 8] = saa_ref
    arr[:, :, 9] = vaa_ref
    arr[:, :, 10] = pressure
    arr[:, :, 11] = aer_model_min
    arr[:, :, 12] = aer_model_max

    # 删除掉nan
    arr_valid = array_simplify.delete_nan(array=arr)  # 经过这步操作，nan会被删除，两个位置参数会加入到最后一行
    deltax = arr_valid[0, :].reshape(1, -1)
    rhoa_nirlx = arr_valid[1, :].reshape(1, -1)
    szax = arr_valid[2, :].reshape(1, -1)
    vzax = arr_valid[3, :].reshape(1, -1)
    saax = arr_valid[4, :].reshape(1, -1)
    vaax = arr_valid[5, :].reshape(1, -1)
    sza_refx = arr_valid[6, :].reshape(1, -1)
    vza_refx = arr_valid[7, :].reshape(1, -1)
    saa_refx = arr_valid[8, :].reshape(1, -1)
    vaa_refx = arr_valid[9, :].reshape(1, -1)
    pressurex = arr_valid[10, :].reshape(1, -1)
    aer_model_minx = arr_valid[11, :].reshape(1, -1)
    aer_model_maxx = arr_valid[12, :].reshape(1, -1)

    if szax.size < 200:
        return None

    global aerosol_models
    aerosol_models = aerosol_models_info

    aero_ref = Aerosol(sza=sza_refx, vza=vza_refx, saa=saa_refx, vaa=vaa_refx)
    # /* compute SS aerosol reflectance in long NIR band */
    rhoas_nirl_min_ref = aero_ref.rhoa_to_rhoas(aermod_index=aer_model_minx, rho_a=rhoa_nirlx, band=bands[nirl_num],
                                                aeroob=1)
    rhoas_nirl_max_ref = aero_ref.rhoa_to_rhoas(aermod_index=aer_model_maxx, rho_a=rhoa_nirlx, band=bands[nirl_num],
                                                aeroob=1)
    phase_min_ref = aero_ref.model_phase(aermod_index=aer_model_minx, band=bands[nirl_num])
    phase_max_ref = aero_ref.model_phase(aermod_index=aer_model_maxx, band=bands[nirl_num])

    aero_tar = Aerosol(sza=szax, vza=vzax, saa=saax, vaa=vaax)
    phase_min_tar = aero_tar.model_phase(aermod_index=aer_model_minx, band=bands[nirl_num])
    phase_max_tar = aero_tar.model_phase(aermod_index=aer_model_maxx, band=bands[nirl_num])

    # 根据散射相函数，将参考传感器的气溶胶单次散射转为目标传感器的单次散射
    rhoas_nirl_max_tar = rhoas_nirl_max_ref * phase_max_tar * np.cos(np.deg2rad(sza_refx)) * np.cos(
        np.deg2rad(vza_refx)) / phase_max_ref / np.cos(np.deg2rad(szax)) / np.cos(np.deg2rad(vzax))
    rhoas_nirl_min_tar = rhoas_nirl_min_ref * phase_min_tar * np.cos(np.deg2rad(sza_refx)) * np.cos(
        np.deg2rad(vza_refx)) / phase_min_ref / np.cos(np.deg2rad(szax)) / np.cos(np.deg2rad(vzax))

    # 计算出模型里所有波长的taua，在这里波段给一个nir_l
    taua_max = aero_tar.model_taua(aermod_index=aer_model_maxx,
                                   rhoas_nir2=rhoas_nirl_max_tar,
                                   nir_l_wave=bands[nirl_num])

    taua_min = aero_tar.model_taua(aermod_index=aer_model_minx,
                                   rhoas_nir2=rhoas_nirl_min_tar,
                                   nir_l_wave=bands[nirl_num])

    taua, t_sensor, t_solar = aero_tar.diff_tran(aer_model_min=aer_model_minx, aer_model_max=aer_model_maxx,
                                                 delta=deltax, pressure=pressurex,
                                                 bands=bands, tauamin=taua_min, tauamax=taua_max, taur=taur)

    # 1 根据确定的模型计算各各波段的epsilon->气溶胶单次散射->气溶胶多次散射
    rhoa = np.full(shape=(bands.size, szax.shape[1]), fill_value=np.nan)
    for num, band in enumerate(bands):
        epsmax = aero_tar.model_epsilon(aermod_index=aer_model_maxx, two_bands=np.array([band, bands[nirl_num]]))
        epsmin = aero_tar.model_epsilon(aermod_index=aer_model_minx, two_bands=np.array([band, bands[nirl_num]]))

        # /* compute SS aerosol reflectance in all bands */
        rhoasmax = rhoas_nirl_max_tar * epsmax
        rhoasmin = rhoas_nirl_min_tar * epsmin

        # /* compute MS aerosol reflectance in visible bands */
        rhoamax = aero_tar.rhoas_to_rhoa(aermod_index=aer_model_maxx, rhoas=rhoasmax,
                                         band=band, aeroob=1)
        rhoamin = aero_tar.rhoas_to_rhoa(aermod_index=aer_model_minx, rhoas=rhoasmin,
                                         band=band, aeroob=1)
        # /* interpolate between upper and lower-bounding models */
        rhoa[num, :] = rhoamin * (1.0 - deltax) + rhoamax * deltax

    rhoa_ = np.full(shape=(bands.size + 2, szax.shape[1]), fill_value=np.nan)
    rhoa_[0:bands.size, :] = rhoa
    rhoa_[-2, :] = arr_valid[-2, :]
    rhoa_[-1, :] = arr_valid[-1, :]
    rhoa_part1 = array_simplify.recover_nan(rows=rhoa_nirl.shape[0], columns=rhoa_nirl.shape[1], new_array=rhoa_)

    F0_t = F0.reshape(1, 1, -1)
    csza = np.zeros(shape=(sza.shape[0], sza.shape[1], 1))
    csza[:, :, 0] = np.cos(np.deg2rad(sza))

    La_part1 = rhoa_part1 * F0_t * csza / np.pi
    # La_part1[rhoa_part1<1e-6]=np.nan

    # 对于近红外波段接近0的点，视为白气溶胶White Aerosol
    # rhoa2小于1e-6视为白气溶胶
    # 重新计算
    # l_a_nir2_part2 = La_part1[:, :, nirl_num] * 1.
    # rhoa2 = rhoa_part1[:, :, nirl_num]
    #
    # rhoa2[np.isnan(rhoa2)] = 999
    # l_a_nir2_part2[rhoa2 > 1e-6] = np.nan
    #
    # # 两部分合到一起，都合并到part1
    # for j in range(bands.size):
    #     La_part2 = l_a_nir2_part2 * (F0_t[0, 0, j] / F0_t[0, 0, nirl_num])  # 具体到波段的白气溶胶
    #     La_part1[:, :, j][~np.isnan(l_a_nir2_part2)] = La_part2[~np.isnan(l_a_nir2_part2)]

    # 漫射透过率和气溶胶光学厚度
    t_ = np.full(shape=(bands.size * 3 + 2, szax.shape[1]), fill_value=np.nan)
    t_[0:bands.size, :] = t_sensor
    t_[bands.size:bands.size * 2, :] = t_solar
    t_[bands.size * 2:bands.size * 3, :] = taua
    t_[-2, :] = arr_valid[-2, :]
    t_[-1, :] = arr_valid[-1, :]
    t_ = array_simplify.recover_nan(rows=sza.shape[0], columns=sza.shape[1], new_array=t_)
    out = {"La": La_part1, "t_sensor": t_[:, :, 0:bands.size], "t_solar": t_[:, :, bands.size:bands.size * 2],
           "taua": t_[:, :, bands.size * 2:bands.size * 3]}
    return out


def atmos_corr(bands=np.array([414., 444, 491, 519, 568, 671, 751, 862]), l_a_nir1=None, l_a_nir2=None, lon=None,
               lat=None, F0=None, sza=None, saa=None, vza=None, vaa=None, nirs_num: int = None, nirl_num: int = None,
               winds_peed=None, pressure=None, relative_humidity=None, taur=None, aerosol_models_info=None):
    """
    Args:
        # nir1表示短波长的近红外波段，nir2表示长波长的近红外波段
        sza:
        F0:
        lat:
        bands:
        l_a_nir1:
        l_a_nir2:
        nirs_num:
        nirl_num:
        aerosol_lut_filepath:
        winds_peed:
        pressure:
        relative_humidity:
        taur:

    Returns:

    """

    # 福亮度转反射率
    rhoa1 = l_a_nir1 * np.pi / F0[nirs_num] / np.cos(np.deg2rad(sza))
    rhoa1[np.isnan(rhoa1)] = -999
    rhoa2 = l_a_nir2 * np.pi / F0[nirl_num] / np.cos(np.deg2rad(sza))
    rhoa2[np.isnan(rhoa2)] = -999
    rhoa2[rhoa2 > predefine.thresholds().cloud] = -999

    # 保证输入的信号正确
    # /* require sufficient signal in two NIR bands *
    rhoamin = 1e-6
    rhoamax = 0.3
    # /* require MS epsilon to be reasonable */
    ms = rhoa1 / rhoa2
    rhoa1[(ms < 0.1) | (ms > 10) | (rhoa1 < rhoamin) | (rhoa1 > rhoamax)] = np.nan
    rhoa2[(ms < 0.1) | (ms > 10) | (rhoa2 < rhoamin) | (rhoa2 > rhoamax)] = np.nan

    # 在进行气溶胶辐射计算之前，删除nan数据，计算完成后，再讲nan放回数组之中，可提高计算效率
    # 把所有的需要的参数都要放到一个数组里
    arr = np.full(shape=(rhoa1.shape[0], rhoa1.shape[1], 11), fill_value=np.nan)
    arr[:, :, 0] = rhoa1
    arr[:, :, 1] = rhoa2
    arr[:, :, 2] = lat
    arr[:, :, 3] = lon
    arr[:, :, 4] = sza
    arr[:, :, 5] = vza
    arr[:, :, 6] = saa
    arr[:, :, 7] = vaa
    arr[:, :, 8] = winds_peed
    arr[:, :, 9] = pressure
    arr[:, :, 10] = relative_humidity

    # 删除掉nan
    arr_valid = array_simplify.delete_nan(array=arr)  # 经过这步操作，nan会被删除，两个位置参数会加入到最后一行
    rhoa1x = arr_valid[0, :].reshape(1, -1)
    rhoa2x = arr_valid[1, :].reshape(1, -1)
    # latx = arr_valid[2, :].reshape(1, -1)
    # lonx = arr_valid[3, :].reshape(1, -1)
    szax = arr_valid[4, :].reshape(1, -1)
    vzax = arr_valid[5, :].reshape(1, -1)
    saax = arr_valid[6, :].reshape(1, -1)
    vaax = arr_valid[7, :].reshape(1, -1)
    # winds_peedx = arr_valid[8, :].reshape(1, -1)
    pressurex = arr_valid[9, :].reshape(1, -1)
    relative_humidityx = arr_valid[10, :].reshape(1, -1)

    # 气溶胶辐射
    print('need to callculate: ' + str(rhoa2x.size) + ' pixels')
    # if rhoa2x.size < predefine.thresholds().pixels_num:
    #     return None
    if rhoa2x.size == 0:
        return None
    global aerosol_models
    aerosol_models = aerosol_models_info
    aero = Aerosol(sza=szax, vza=vzax, saa=saax, vaa=vaax)
    result_temp = aero.wang_method(rho_a_nir1=rhoa1x, rho_a_nir2=rhoa2x, pressure=pressurex, bands=bands, taur=taur,
                                   relative_humidity=relative_humidityx, nirs_num=nirs_num, nirl_num=nirl_num)
    rhoa, taua, t_sensor, t_solar, aer_model_min, aer_model_max, delta, rhoasmin_nir2, rhoasmax_nir2 = result_temp

    # 多次散射反射率转回福亮度
    F0_t = F0.reshape(-1, 1)
    la2_temp1 = rhoa / (np.pi / F0_t / np.cos(szax * np.pi / 180))
    La2_ = np.full(shape=(bands.size + 2, la2_temp1.shape[1]), fill_value=np.nan)
    La2_[0:bands.size, :] = la2_temp1
    La2_[-2, :] = arr_valid[-2, :]
    La2_[-1, :] = arr_valid[-1, :]

    La2_part1 = array_simplify.recover_nan(rows=l_a_nir1.shape[0], columns=l_a_nir1.shape[1], new_array=La2_)

    # 对于近红外波段接近0的点，视为白气溶胶White Aerosol
    # rhoa2小于1e-6视为白气溶胶
    # 重新计算
    rhoa2 = l_a_nir2 * np.pi / F0[-1] / np.cos(sza * np.pi / 180)
    l_a_nir2_part2 = l_a_nir2 * 1.
    rhoa2[np.isnan(rhoa2)] = 999
    l_a_nir2_part2[rhoa2 > 1e-6] = np.nan

    # 两部分合到一起，都合并到part1
    for j in range(bands.size):
        La2_part2 = l_a_nir2_part2 * (F0_t[j] / F0_t[-1])  # 具体到波段的白气溶胶
        La2_part1[:, :, j][La2_part1[:, :, j] == np.nan] = La2_part2[La2_part1[:, :, j] == np.nan]
        # La2_part1[:, :, j][rhoa2 < 1e-6] = np.nan
        # La2_part1[:, :, j] = np.nanmean(np.dstack([La2_part1[:, :, j], La2_part2[:, :, j]]), axis=2)

    # 漫射透过率和气溶胶光学厚度
    t_ = np.full(shape=(bands.size * 3 + 2, la2_temp1.shape[1]), fill_value=np.nan)
    t_[0:bands.size, :] = t_sensor
    t_[bands.size:bands.size * 2, :] = t_solar
    t_[bands.size * 2:bands.size * 3, :] = taua
    t_[-2, :] = arr_valid[-2, :]
    t_[-1, :] = arr_valid[-1, :]
    t_ = array_simplify.recover_nan(rows=l_a_nir1.shape[0], columns=l_a_nir1.shape[1], new_array=t_)

    # 气溶胶模型
    aer = np.full(shape=(3 + 2, la2_temp1.shape[1]), fill_value=np.nan)
    aer[0, :] = delta
    aer[1, :] = rhoasmin_nir2
    aer[2, :] = rhoasmax_nir2
    aer[-2, :] = arr_valid[-2, :]
    aer[-1, :] = arr_valid[-1, :]
    aer1 = array_simplify.recover_nan(rows=l_a_nir1.shape[0], columns=l_a_nir1.shape[1], new_array=aer)
    del aer

    aer = np.full(shape=(2 + 2, la2_temp1.shape[1]), fill_value=np.nan)
    aer[0, :] = aer_model_min * 1.
    aer[1, :] = aer_model_max * 1.
    aer[2, :] = arr_valid[-2, :]
    aer[3, :] = arr_valid[-1, :]
    aer2 = array_simplify.recover_nan(rows=l_a_nir1.shape[0], columns=l_a_nir1.shape[1], new_array=aer)

    return (La2_part1, t_[:, :, 0:bands.size], t_[:, :, bands.size:bands.size * 2],
            t_[:, :, bands.size * 2:bands.size * 3], aer1, aer2)

# def fixdaot(aot: np.ndarray = None, relative_humidity=None, sza=None, vza=None, saa=None, vaa=None, pressure=None,
#             taur=None, iwnir_s=6, iwnir_l=7, aerosol_lut_filepath=None,
#             wave: np.ndarray = np.array([412, 443, 490, 520, 565, 670, 750, 865])):
#     """
#     根据aot计算气溶胶辐亮度
#     aot shape：
#           band1   band2   .。。。bandx
#     loc1    xx1     xx2          xx3
#     loc2    xx4     xx5          xx6
#     .
#     locn   xx7      xx8          xx9
#     /* -
#     --------------------------------------------------------------------------------------- */
#     /* fixedaot() - compute aerosol reflectance for fixed aot(lambda)                           */
#     /* B. Franz, August 2004.                                                                   */
#     /* ---------------------------------------------------------------------------------------- */
#     :return:
#     """
#
#     """
#     /* compute angstrom and use to select bounding models */
#     """
#     # 找出两波段的位置
#     angst_band1 = np.argmin(np.abs(520 - wave))
#     angst_band2 = np.argmin(np.abs(865 - wave))
#     # 计算angstrom系数
#     angstrom = -np.log(aot[:, :, angst_band1] / aot[:, :, angst_band2]) / np.log(wave[angst_band1] / wave[angst_band2])
#     angstrom[aot[:, :, angst_band2] <= 0] = 0.
#     # 1读取查找表
#     print('====loading aerosol LUT ...')
#     global aerosol_models
#     aerosol_models = load_aermod(aerosol_lut_filepath=aerosol_lut_filepath)
#
#     aer_model_min, aer_model_max, ang_mod_low, ang_mod_up, delta = model_select_angstrom(angstrom_measure=angstrom,
#                                                                                          relative_humidity=relative_humidity,
#                                                                                          band520=wave[angst_band1])
#
#     # /* compute factor for SS approximation, set-up for interpolation */
#     # # /* get model phase/albedo function for all wavelengths at this geometry for the two models */
#     reaa = saa - vaa
#     reaa = np.abs(reaa)
#     reaa[reaa > 180.] = reaa[reaa > 180.] - 180
#     rhoa = np.full(shape=(wave.size, ang_mod_up.size), fill_value=np.nan)
#
#     for i, band_i in enumerate(wave):
#         phase_min = model_phase(aermod_index=aer_model_min, sza=sza, saa=saa, vza=vza, vaa=vaa, band=band_i)
#         phase_max = model_phase(aermod_index=aer_model_max, sza=sza, saa=saa, vza=vza, vaa=vaa, band=band_i)
#         albedo_min = model_albedo(aermod_index=aer_model_min, band=band_i)
#         albedo_max = model_albedo(aermod_index=aer_model_max, band=band_i)
#         f_min = albedo_min * phase_min / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#         f_max = albedo_max * phase_max / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#         # 如果查找表的波长与观测的aot的波长不是对应的，则需要插值，使用log变换后再插值.在这里未写插值过程
#         lnf_min = np.log(f_min)
#         lnff_max = np.log(f_max)
#
#         # 单次散射
#         rhoas_min = aot[:, :, i] * f_min
#         rhoas_max = aot[:, :, i] * f_max
#         if i == iwnir_s:
#             rhoas_min_nirs = rhoas_min
#             rhoas_max_nirs = rhoas_max
#         if i == iwnir_l:
#             rhoas_min_nirl = rhoas_min
#             rhoas_max_nirl = rhoas_max
#
#         # 多次散射/* compute MS aerosol reflectance in visible bands */
#         rhoamin = rhoas_to_rhoa(aermod_index=aer_model_min, rhoas=rhoas_min, sza=sza, vza=vza, saa=saa, vaa=vaa,
#                                 band=np.array([band_i]), aeroob=1)
#         rhoamax = rhoas_to_rhoa(aermod_index=aer_model_max, rhoas=rhoas_max, sza=sza, vza=vza, saa=saa, vaa=vaa,
#                                 band=np.array([band_i]), aeroob=1)
#         rhoa[i, :] = (1.0 - delta) * rhoamin + delta * rhoamax
#
#     eps_min = rhoas_min_nirs / rhoas_min_nirl
#     eps_max = rhoas_max_nirs / rhoas_max_nirl
#     epsnir = (1.0 - delta) * eps_min + delta * eps_max
#
#     # 计算出模型里所有波长的taua，在这里波段给一个nir_l
#     taua_max = model_taua(aermod_index=aer_model_max, sza=sza, saa=saa, vza=vza, vaa=vaa, rhoas_nir2=rhoas_max_nirl,
#                           nir_l_wave=wave[iwnir_l])
#     taua_min = model_taua(aermod_index=aer_model_min, sza=sza, saa=saa, vza=vza, vaa=vaa, rhoas_nir2=rhoas_min_nirl,
#                           nir_l_wave=wave[iwnir_l])
#
#     taua, t_sensor, t_solar = diff_tran(aer_model_min=aer_model_min, aer_model_max=aer_model_max,
#                                         delta=delta, sza=sza, vza=vza, pressure=pressure,
#                                         bands=wave, tauamin=taua_min, tauamax=taua_max, taur=taur)
#
#     return rhoa, epsnir, t_sensor, t_solar
#
# def fixd_aer_aot(aot: np.ndarray, modnum, taur, pressure, sza, vza, reaa, aerosol_lut_filepath,
#                  wave: np.ndarray = np.array([412, 443, 490, 520, 565, 670, 750, 865])):
#     """
#     aot为第二个近红外波段的的气溶胶光学厚度
#     给定气溶胶光学厚度和气溶胶模型，模拟气溶胶反射率
#     :return:
#     """
#     global aerosol_models
#     aerosol_models = load_aermod(aerosol_lut_filepath=aerosol_lut_filepath)
#     rhoa = np.full(shape=(aot.shape[0], aot.shape[1], wave.size), fill_value=np.nan)
#
#     # 计算865的单次散射
#     phase_nirl = model_phase(aermod_index=modnum, sza=sza, vza=vza, reaa=reaa, band=wave[-1])
#     albedo_nirl = model_albedo(aermod_index=modnum, band=wave[-1])
#     f_nirl = albedo_nirl * phase_nirl / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#     rhoas_nirl = aot[:, :, -1] * f_nirl
#
#     taua = model_taua(aermod_index=modnum, sza=sza, vza=vza, reaa=reaa, rhoas_nir2=rhoas_nirl,
#                       nir_l_wave=wave[-1])
#
#     for i, band_i in enumerate(wave):
#         phase = model_phase(aermod_index=modnum, sza=sza, vza=vza, reaa=reaa, band=band_i)
#         albedo = model_albedo(aermod_index=modnum, band=band_i)
#         f = albedo * phase / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#
#         # 如果查找表的波长与观测的aot的波长不是对应的，则需要插值，使用log变换后再插值.在这里未写插值过程
#         lnf = np.log(f)
#
#         # 单次散射
#         rhoas = taua[i, :] * f
#
#         # 多次散射/* compute MS aerosol reflectance in visible bands */
#         rhoa_temp = rhoas_to_rhoa(aermod_index=modnum, rhoas=rhoas, sza=sza, vza=vza, reaa=reaa,
#                                   band=np.array([band_i]), aeroob=1)
#         rhoa[:, :, i] = rhoa_temp
#
#     aot_in = taua
#     taua_temp, t_sensor, t_solar = diff_tran(aer_model_min=modnum, aer_model_max=modnum,
#                                              delta=0, sza=sza, vza=vza, pressure=pressure,
#                                              bands=wave, tauamin=aot_in, tauamax=aot_in, taur=taur)
#
#     return taua, rhoa, t_sensor, t_solar
#
# def fixed_2bands_aot(aot: np.ndarray = None, relative_humidity=None, sza=None, vza=None, saa=None, vaa=None,
#                      pressure=None, taur=None,aerosol_lut_filepath=None,
#                      wave: np.ndarray = np.array([412, 443, 490, 520, 565, 670, 750, 865]), iwnir_s=6, iwnir_l=7):
#     """
#     仅仅知道两个波段的气溶胶光学厚度
#     Returns:
#     """
#     # 找出两波段的位置
#     angst_band1 = np.argmin(np.abs(520 - wave))
#     angst_band2 = np.argmin(np.abs(865 - wave))
#     # 计算angstrom系数
#     angstrom = -np.log(aot[:, :, angst_band1] / aot[:, :, angst_band2]) / np.log(wave[angst_band1] / wave[angst_band2])
#     angstrom[aot[:, :, angst_band2] <= 0] = 0.
#     # 1读取查找表
#     print('====loading aerosol LUT ...')
#     global aerosol_models
#     aerosol_models = load_aermod(aerosol_lut_filepath=aerosol_lut_filepath)
#
#     aer_model_min, aer_model_max, ang_mod_low, ang_mod_up, delta = model_select_angstrom(angstrom_measure=angstrom,
#                                                                                          relative_humidity=relative_humidity,
#                                                                                          band520=wave[angst_band1])
#     # /* compute factor for SS approximation, set-up for interpolation */
#     # # /* get model phase/albedo function for all wavelengths at this geometry for the two models */
#     reaa = saa - vaa
#     reaa = np.abs(reaa)
#     reaa[reaa > 180.] = reaa[reaa > 180.] - 180
#     rhoa = np.full(shape=(wave.size, ang_mod_up.size), fill_value=np.nan)
#     # 计算865的单次散射
#     phase_nirl_min = model_phase(aermod_index=aer_model_min, sza=sza, vza=vza, reaa=reaa, band=wave[-1])
#     albedo_nirl_min = model_albedo(aermod_index=aer_model_min, band=wave[-1])
#     f_nirl_min = albedo_nirl_min * phase_nirl_min / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#     rhoas_nirl_min = aot[:, :, -1] * f_nirl_min
#     taua_min = model_taua(aermod_index=aer_model_min, sza=sza, vza=vza, reaa=reaa, rhoas_nir2=rhoas_nirl_min,
#                           nir_l_wave=wave[-1])
#
#     phase_nirl_max = model_phase(aermod_index=aer_model_max, sza=sza, vza=vza, reaa=reaa, band=wave[-1])
#     albedo_nirl_max = model_albedo(aermod_index=aer_model_max, band=wave[-1])
#     f_nirl_max = albedo_nirl_max * phase_nirl_max / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#     rhoas_nirl_max = aot[:, :, -1] * f_nirl_max
#     taua_max = model_taua(aermod_index=aer_model_max, sza=sza, vza=vza, reaa=reaa, rhoas_nir2=rhoas_nirl_max,
#                           nir_l_wave=wave[-1])
#
#     for i, band_i in enumerate(wave):
#         phase_min = model_phase(aermod_index=aer_model_min, sza=sza, vza=vza, reaa=reaa, band=band_i)
#         albedo_min = model_albedo(aermod_index=aer_model_min, band=band_i)
#         f_min = albedo_min * phase_min / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#         # 如果查找表的波长与观测的aot的波长不是对应的，则需要插值，使用log变换后再插值.在这里未写插值过程
#         lnf_min = np.log(f_min)
#         # 单次散射
#         rhoas = taua_min[i, :] * f_min
#         # 多次散射/* compute MS aerosol reflectance in visible bands */
#         rhoa_min_temp = rhoas_to_rhoa(aermod_index=aer_model_min, rhoas=rhoas, sza=sza, vza=vza, reaa=reaa,
#                                       band=np.array([band_i]), aeroob=1)
#
#         phase_max = model_phase(aermod_index=aer_model_max, sza=sza, vza=vza, reaa=reaa, band=band_i)
#         albedo_max = model_albedo(aermod_index=aer_model_max, band=band_i)
#         f_max = albedo_max * phase_max / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
#         # 如果查找表的波长与观测的aot的波长不是对应的，则需要插值，使用log变换后再插值.在这里未写插值过程
#         lnf_max = np.log(f_max)
#         # 单次散射
#         rhoas = taua_max[i, :] * f_max
#         # 多次散射/* compute MS aerosol reflectance in visible bands */
#         rhoa_max_temp = rhoas_to_rhoa(aermod_index=aer_model_max, rhoas=rhoas, sza=sza, vza=vza, reaa=reaa,
#                                       band=np.array([band_i]), aeroob=1)
#
#         rhoa[i, :] = (1.0 - delta) * rhoa_min_temp + delta * rhoa_max_temp
#
#     taua_temp, t_sensor, t_solar = diff_tran(aer_model_min=aer_model_min, aer_model_max=aer_model_max,
#                                              delta=0, sza=sza, vza=vza, pressure=pressure,
#                                              bands=wave, tauamin=taua_min, tauamax=taua_max, taur=taur)
#     taua = (1.0 - delta) * taua_min + delta * taua_max
#     return taua, rhoa, t_sensor, t_solar
