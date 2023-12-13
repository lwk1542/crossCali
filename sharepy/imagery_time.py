# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/4 21:41
@FileName: imagery_time.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import datetime


def obtain_time(sensor:str, file):

    if any([sensor == 's3a_olci',sensor == 's3b_olci']):
        date_str = os.path.basename(file)[16:31]
        print("time:{}".format(date_str))
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]
        date_str2 = date_str[0:8]
        doy = datetime.datetime.strptime(date_str, '%Y%m%dT%H%M%S').strftime('%j')
    elif any([sensor == 'terra_modis', sensor == 'aqua_modis']):
        date_str = os.path.basename(file)[10:17]  # [10:17] MOD021KM.A2021332.2335.061._seadas_rrs.hdf
        year, doy = date_str[0:4], date_str[4:7]
        date = datetime.datetime.strptime(year + doy, '%Y%j')
        date_str2 = date.strftime('%Y%m%d')
        month, day = date_str2[4:6], date_str2[6:8]

    return year, month, day, doy, date_str2
