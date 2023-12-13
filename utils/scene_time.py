# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/1 14:43
@FileName: time_location.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import datetime


def scene_time(file):
    if "KX10_MII_" in file:
        date_str = os.path.basename(file).split('_')[2]
        year, month, day = date_str[0:4], date_str[4:6], date_str[6:]
        doy = datetime.datetime.strptime(date_str, '%Y%m%d').strftime('%j')
    return date_str  # , year, month, day, doy