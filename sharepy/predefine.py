# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: predefine.py
@time: 2021/3/25 10:22
@desc:
"""


def thresholds():
    thresholds.aot = 1.
    thresholds.chl = 100.
    thresholds.cloud = 0.2
    # thresholds.cloud_nl = 0.3
    thresholds.cloudwave = 865
    thresholds.epsmax = 1.35
    thresholds.epsmin = 0.8
    thresholds.glint_threshold = 0.005
    thresholds.ice_threshlod = 0.1
    thresholds.nlw_min = 0.15
    thresholds.rhoa_min = 0.0002
    thresholds.vza_max = 60.0
    thresholds.sza_max = 70.
    thresholds.taua_max = 0.3
    thresholds.windspeed_max = 12
    thresholds.glint_iter_max = 1 # 从0次起算，1代表2次
    thresholds.pixels_num=500 # 不到规定的有效像元数，该景影像不再计算
    thresholds.seed_chl=0.
    thresholds.seed_green = 0.
    thresholds.seed_red = 0.

    thresholds.chlmin = 0.001
    thresholds.chlmax = 1000.0
    thresholds.chlbad = -32767.
    thresholds.BAD_FLT = -32767.
    thresholds.minrat = 0.21
    thresholds.maxrat = 30.0
    thresholds.brdf_maxiter = 3
    thresholds.df = 0.33
    # seadas 界面默认
    thresholds.aer_iter_max = 1#3#10
    # seadas程序代码
    thresholds.aer_iter_min = 0#2

    thresholds.cbot = 0.7
    thresholds.ctop = 1.3
    thresholds.nir_chg = 0.02

    return thresholds


if __name__ == '__main__':
    a = thresholds().glint_threshold
    print(a)
