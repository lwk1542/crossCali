# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: get_chl.py
@time: 2021/6/15 15:39
@desc: 计算叶绿素浓度，目前使用OCI算法，完成
"""
import numpy as np

# from nptyping import Array
from sharepy import predefine


def get_default_chl(rrs: np.ndarray, bands: np.ndarray, b443: int, b490: int, b520: int, b555: int,
                    b670: int, sensorid: str):
    match sensorid:
        case "hy1ccocts":
            from sensor.hy1ccocts import retrivel
            chl = retrivel.chl(rrs, bands, b443, b555, b670).value()
        case "hy1dcocts":
            # a = np.array([0.3272, -2.9940, 2.7218, -1.2259, -0.5683])  #  seawifs
            # 直接使用了oci算法， 该算法也是seadas的默认算法
            from sensor.hy1ccocts import retrivel
            chl = chl_oci(rrs=rrs, bands=bands, b443=b443, b490=b490, b520=b520, b555=b555, b670=b670)
        case _:
            #  直接使用了oci算法， 该算法也是seadas的默认算法
            chl = chl_oci(rrs=rrs, bands=bands, b443=b443, b490=b490, b520=b520, b555=b555, b670=b670)
    return chl


def chl_oci(rrs: np.ndarray = None, bands: np.ndarray=None, b443: int = None, b490: int = None, b520: int = None, b555: int = None, b670: int = None):
    """
    完成
    Args:
        rrs ():
        b443 ():
        b490 ():
        b520 ():
        b555 ():
        b670 ():

    Returns:

    """
    t1 = 0.15
    t2 = 0.20
    chl1 = chl_hu(rrs=rrs, bands=bands, b443=b443, b555=b555, b670=b670)
    chl2 = chl_oc4(rrs=rrs, b443=b443, b490=b490, b520=b520, b565=b555)
    chl = np.full_like(rrs[:, :, 0], fill_value=predefine.thresholds().chlbad)
    chl[chl1 < t1] = chl1[chl1 < t1]
    chl[chl1 > t2] = chl2[chl1 > t2]
    # chl = chl1 * (t2 - chl1) / (t2 - t1) + chl2 * (chl1 - t1) / (t2 - t1);
    ju = (chl1 > t1) & (chl1 < t2) & (chl2 > 0)
    chl[ju] = chl1[ju] * (t2 - chl1[ju]) / (t2 - t1) + chl2[ju] * (chl1[ju] - t1) / (t2 - t1)
    return chl


def chl_hu(rrs: np.ndarray = None, bands: np.ndarray = None, b443: int = None, b555: int = None,
           b670: int = None):
    """
    完成
    Args:
        rrs ():
        bands ():
        b443 ():
        b555 ():
        b670 ():

    Returns:

    """
    w = np.array([443, 555, 670])
    c = np.array([-0.4909, 191.6590])
    rrs1 = rrs[:, :, b443]
    rrs2 = rrs[:, :, b555]
    rrs3 = rrs[:, :, b670]
    # 如果波段不是555，需要进行一定的调整,比如565
    rrs2 = conv_rrs_to_555(Rrs=rrs2, wave=bands[b555])
    # // compute index
    ci = np.full_like(rrs[:, :, 0], fill_value=np.nan)
    ju = (rrs3 > predefine.thresholds().BAD_FLT + 1) & (rrs2 > predefine.thresholds().BAD_FLT + 1) & (
            rrs1 > predefine.thresholds().BAD_FLT + 1)
    ci[ju] = 0
    temp = rrs2 - (rrs1 + (w[1] - w[0]) / (w[2] - w[0]) * (rrs3 - rrs1))
    ju1 = (rrs3 > predefine.thresholds().BAD_FLT + 1) & (rrs2 > 0.0) & (rrs1 > 0.0) & (temp < 0.0)
    ci[ju1] = temp[ju1]

    # // index should be negative in algorithm-validity range
    ci[ci > 0] = 0.

    chl = np.power(10.0, c[0] + c[1] * ci)
    chl[chl < predefine.thresholds().chlmin] = predefine.thresholds().chlmin
    chl[chl > predefine.thresholds().chlmax] = predefine.thresholds().chlmax
    return chl


def chl_oc4(rrs: np.ndarray, b443: int, b490: int, b520: int, b565: int , a:list):

    if not b520:
        b520 = b490
    rrs1 = rrs[:, :, b443]
    rrs2 = rrs[:, :, b490]
    rrs3 = rrs[:, :, b520]
    rrs4 = rrs[:, :, b565]
    rat = np.fmax.reduce([rrs1, rrs2, rrs3]) / rrs4

    chl = np.power(10.0, (a[0] + rat * (a[1] + rat * (a[2] + rat * (a[3] + rat * a[4])))))
    chl[chl < predefine.thresholds().chlmin] = predefine.thresholds().chlmin
    chl[chl > predefine.thresholds().chlmax] = predefine.thresholds().chlmax
    return chl


def conv_rrs_to_555(Rrs: np.ndarray = None, wave: np.ndarray = None):
    convert = 1
    if np.abs(wave - 555) > 2:
        sw = 0.001723
        a1 = 0.986
        b1 = 0.081495
        a2 = 1.031
        b2 = 0.000216
    elif np.abs(wave - 550) <= 2:
        sw = 0.001597
        a1 = 0.988
        b1 = 0.062195
        a2 = 1.014
        b2 = 0.000128
    elif np.abs(wave - 560) <= 2:
        sw = 0.001148
        a1 = 1.023
        b1 = -0.103624
        a2 = 0.979
        b2 = -0.000121
    elif np.abs(wave - 550) <= 3:
        # 为了SDG MII设置的这个选项，这个设置并不对，只是沿用了<=2的情况
        sw = 0.001597
        a1 = 0.988
        b1 = 0.062195
        a2 = 1.014
        b2 = 0.000128
    elif np.abs(wave - 565) <= 2:
        sw = 0.000891
        a1 = 1.039
        b1 = -0.183044
        a2 = 0.971
        b2 = -0.000170
    else:
        convert = 0
        sw = None
        a1 = None
        b1 = None
        a2 = None
        b2 = None

    if convert == 0:
        print(" Unable to convert Rrs at %f to 555nm")
        return Rrs
    else:
        Rrs[Rrs < sw] = np.power(10.0, a1 * np.log10(Rrs[Rrs < sw]) - b1)
        Rrs[Rrs >= sw] = a2 * Rrs[Rrs >= sw] - b2
        return Rrs




