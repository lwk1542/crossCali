# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: gas_transmittance.py
@time: 2021/3/5 10:59
@desc: diffuse transmission by ozone/NO2
   《Atmospheric Correction for Satellite Ocean Color Radiometry》Curtis Mobley
"""
import numpy as np


# seadas里面使用校正因子取代了氧气吸收计算，
# /* ------------------------------------------------------------------- */
# /* correction factor to replace oxygen absorption to Lr(765)           */
#
# /* ------------------------------------------------------------------- */
def oxygen_ray(airmass=None):
    # /* base case is the 1976 Standard atmosphere without aerosols */
    # 瑞利的氧气吸收
    a = np.array([-1.3491, 0.1155, -7.0218e-3])
    return 1.0 / (1.0 + np.power(10.0, a[0] + airmass * a[1] + airmass * airmass * a[2]))


# /* ------------------------------------------------------------------- */
# /* correction factor to remove oxygen absorption from La(765)          */
#
# /* ------------------------------------------------------------------- */
def oxygen_aer(airmass=None):
    # /* base case: m80_t50_strato  visibility (550nm ,0-2km):25km */
    # 气溶胶的氧气吸收
    a = np.array([-1.0796, 9.0481e-2, -6.8452e-3])
    return 1.0 + np.power(10.0, a[0] + airmass * a[1] + airmass * airmass * a[2])


def transmittance_o2(sza=None, vza=None, koz: float = None, concentration=None):
    """
    Args:
        sza ():
        vza ():
        koz (): 臭氧吸收截面 absorption cross section (in cm2 molecule−1)
        concentration ():臭氧浓度 column amount in molecules cm−2

    Returns:

    """

    mu = sza * np.pi / 180
    mu0 = vza * np.pi / 180
    tau_o3 = concentration.reshape(concentration.shape[0], concentration.shape[1], 1) * koz.reshape(1, 1, -1)
    tg_solar_o3 = np.exp(-tau_o3 / (np.cos(mu.reshape(mu.shape[0], mu.shape[1], 1))))
    tg_sensor_o3 = np.exp(-tau_o3 / (np.cos(mu0.reshape(mu0.shape[0], mu0.shape[1], 1))))

    return tg_sensor_o3, tg_solar_o3


def transmittance_o3(sza=None, vza=None, koz: float = None, concentration=None):
    """
    Args:
        sza ():
        vza ():
        koz (): 臭氧吸收截面 absorption cross section (in cm2 molecule−1)
        concentration ():臭氧浓度 column amount in molecules cm−2

    Returns:

    """

    mu = sza * np.pi / 180
    mu0 = vza * np.pi / 180
    tau_o3 = concentration.reshape(concentration.shape[0], concentration.shape[1], 1) * koz.reshape(1, 1, -1)
    tg_solar_o3 = np.exp(-tau_o3 / (np.cos(mu.reshape(mu.shape[0], mu.shape[1], 1))))
    tg_sensor_o3 = np.exp(-tau_o3 / (np.cos(mu0.reshape(mu0.shape[0], mu0.shape[1], 1))))

    return tg_sensor_o3, tg_solar_o3


def transmittance_NO2(kno2: float = None, sza=None, vza=None, strat_no2=None, trop_no2=None):
    """
    Args:
        trop_no2:
        strat_no2:
        kno2:
        sza ():
        vza ():
        k (): 吸收截面
        concentration ():浓度

    Returns:

    """
    # if sensor=='h1c_cocts':
    #     koz, kno2 =satellite_sensor.k_gas_h1c_cocts()
    no2_tr200 = np.zeros_like(sza)
    no2_tr200[trop_no2 > 0] = np.exp(12.6615 + 0.61676 * np.log(trop_no2[trop_no2 > 0]))
    a_285 = kno2.reshape(1, 1, -1) * (1.0 - 0.003 * (285.0 - 294.0))
    a_225 = kno2.reshape(1, 1, -1) * (1.0 - 0.003 * (225.0 - 294.0))
    tau_to200 = a_285 * no2_tr200.reshape(sza.shape[0], sza.shape[1], 1) + a_225 * strat_no2.reshape(sza.shape[0],
                                                                                                     sza.shape[1], 1)
    mu = sza * np.pi / 180
    mu0 = vza * np.pi / 180
    tg_solar_no2 = np.exp(tau_to200 * (1 / (np.cos(mu.reshape(mu.shape[0], mu.shape[1], 1)))))
    tg_sensor_no2 = np.exp(tau_to200 * (1 / (np.cos(mu0.reshape(mu0.shape[0], mu0.shape[1], 1)))))
    return tg_sensor_no2, tg_solar_no2


def transmittance_co2(sza=None, vza=None, t_co2=None):
    mu = sza * np.pi / 180
    mu0 = vza * np.pi / 180
    t_co2 = t_co2.reshape(1, 1, -1)
    tg_sol = np.power(t_co2, 1.0 / (np.cos(mu.reshape(mu.shape[0], mu.shape[1], 1))))
    tg_sen = np.power(t_co2, 1.0 / (np.cos(mu0.reshape(mu0.shape[0], mu0.shape[1], 1))))
    return tg_sen, tg_sol


def transmittance_h2o(sza=None, vza=None, zia_table=None, water_vapor=None):
    [a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o] = zia_table
    a_h2o = a_h2o.reshape(1, 1, -1)
    b_h2o = b_h2o.reshape(1, 1, -1)
    c_h2o = c_h2o.reshape(1, 1, -1)
    d_h2o = d_h2o.reshape(1, 1, -1)
    e_h2o = e_h2o.reshape(1, 1, -1)
    f_h2o = f_h2o.reshape(1, 1, -1)
    g_h2o = g_h2o.reshape(1, 1, -1)
    water_vapor = water_vapor.reshape(water_vapor.shape[0], water_vapor.shape[1], 1)
    t_h2o = a_h2o + water_vapor * (b_h2o + water_vapor * (
                c_h2o + water_vapor * (d_h2o + water_vapor * (e_h2o + water_vapor * (f_h2o + water_vapor * g_h2o)))))
    mu = sza * np.pi / 180
    mu0 = vza * np.pi / 180

    tg_sol = np.power(t_h2o, 1.0 / (np.cos(mu.reshape(mu.shape[0], mu.shape[1], 1))))
    tg_sen = np.power(t_h2o, 1.0 / (np.cos(mu0.reshape(mu0.shape[0], mu0.shape[1], 1))))
    return tg_sen, tg_sol


def tg():
    transmittance_o3(sza=None, vza=None, sensor=None, concentration=None)


class satellite_sensor:
    @staticmethod
    def k_gas_h1c_cocts():
        # 衰减系数cm-1
        # koz = np.array([0.0007518062000084764, 0.003246272994972116, 0.022962366914853878, 0.04971195735341498, 0.11748533851298386,
        #        0.044307962060693776, 0.009407191725330917, 0.0021894200795084364])
        # kno2 = np.array([5.9749371999315555e-19, 4.996484866210154e-19, 2.7730882317533164e-19, 1.8104308043033148e-19,
        #         7.199608925044155e-20, 7.80682087019753e-21, 1.1618567864144775e-21, 5.121495997604189e-23])
        kno2 = np.array([5.964517378098426e-19, 5.01667526749566e-19, 2.7848945048908554e-19, 1.802352368086221e-19,
                         7.184451458522253e-20, 7.79749419156795e-21, 1.1646586908876065e-21, 5.0782980858941893e-23])

        # 消光截面cm2粒子数
        koz = np.array([5.383974200705576e-23, 1.6887334980611554e-22, 9.286214356410431e-22, 1.931216823979986e-21,
                        4.494106553398089e-21, 1.769899601233302e-21, 3.9318364922493383e-22, 8.995093366407616e-23])
        return koz, kno2
