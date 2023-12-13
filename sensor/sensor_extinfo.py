# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/25 21:14
@FileName: sensor_extinfo.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""


def satellite_alt(sensor_id):
    if sensor_id == "sdgsat1mii":
        alt = 505
    elif sensor_id == "olcis3a":
        alt = 750
    elif sensor_id == "olcis3b":
        alt = 750
    else:
        alt = 800
    return alt