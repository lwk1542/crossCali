# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: airmass.py
@time: 2021/3/13 21:10
@desc:
    zh.wikipedia.org/wiki/大气质量_(天文学)
    https://oceancolor.gsfc.nasa.gov/docs/ocssw/airmass_8c_source.html
"""
import numpy as np


#  airmass for a spherical atmosphere, Kasten & Young, 1989
def ky_airmass(zenithangle=None):
    mu = np.cos(zenithangle * np.pi / 180)
    mu[mu < 0.01] = 0.01
    return 1. / (mu + np.power(0.50572 * (96.07995 - zenithangle), (-1.6364)))


#  airmass for a plane parallel atmosphere
def pp_airmass(zenithangle=None):
    mu = np.cos(zenithangle * np.pi / 180)
    mu[mu < 0.01] = 0.01
    return 1. / mu
