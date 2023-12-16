# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/29 14:44
@FileName: atmoscorr_v0.1.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os
import main_ac

# path =""

sensorids = ["fy3dmersi", "hy1ccocts", "hy1dcocts"]
ac = main_ac.Calcu(filespath=r"E:\XDA\fy3dmersi",
                   sensorid="fy3dmersi")
ac.run_main()
