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
from pathlib import Path

def scene_time(file_dir):
    if isinstance(file_dir, Path):
        #print("这是一个 pathlib 对象")
        file_str = str(file_dir)  # 转换为字符串
    else:
        file_str = file_dir

    if "KX10_MII_" in file_str:
        date_str = os.path.basename(file_str).split('_')[2]
        year, month, day = date_str[0:4], date_str[4:6], date_str[6:]
        doy = datetime.datetime.strptime(date_str, '%Y%m%d').strftime('%j')
        
    elif 'H1D' in file_str:
        target_file_date = os.path.splitext(os.path.basename(file_str))[0]
        # 格式: H1D_OPER_OCT_L1B_20210102T013000_20210102T013500_02952_10.h5
        date_str = target_file_date.split('_')[4]
        # 使用下划线分割: 20210102T013000 -> 2021_01_02T01_30_00
        # year = date_part[0:4]
        # month = date_part[4:6]
        # day = date_part[6:8]
        # hour = date_part[9:11]
        # minute = date_part[11:13]
        # second = date_part[13:15]
        # date_str = f"{year}_{month}_{day}_{hour}_{minute}_{second}"
    return date_str#, year, month, day, None

    