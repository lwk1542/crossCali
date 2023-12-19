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

path=r"E:\XDA\fy3dmersi"

sensorids = ["fy3dmersi", "hy1ccocts", "hy1dcocts"]
sdgsat1mii_para = {"block_size_rows": 150, "rrs_out": False, "sensorid": "sdgsat1mii", "rrc_out": True,
                   "filespath":r"G:\SDGsat\calibration\sea\2023\validation\turbid\target"}
hy1ccocts_para = {"block_size_rows": 5000, "rrs_out": True, "sensorid": "hy1ccocts", "rrc_out": True,
                   "filespath":r"G:\SDGsat\calibration\sea\2023\validation\turbid\comp"}

para = hy1ccocts_para

ac = main_ac.Calcu(filespath=para["filespath"],
                   sensorid=para["sensorid"],
                   block_size_rows=para["block_size_rows"],
                   rrs_out=para["rrs_out"],
                   rrc_out=para["rrc_out"])
ac.run_main()


