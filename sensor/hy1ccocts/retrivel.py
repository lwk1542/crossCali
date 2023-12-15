# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/12 15:53
@FileName: retrivel.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np


class chl(object):
    def __init__(self, rrs: np.ndarray = None, bands: np.ndarray = None, b443: int = None, b555: int = None,
                 b670: int = None):
        # # X. Ye, J. Liu, M. Lin, J. Ding, B. Zou and Q. Song, "Global Ocean Chlorophyll-a Concentrations Derived
        # From COCTS Onboard the HY-1C Satellite and Their Preliminary Evaluation," in IEEE Transactions on Geoscience
        # and Remote Sensing, vol. 59, no. 12, pp. 9914-9926, Dec. 2021, doi: 10.1109/TGRS.2020.3036963.
        self.rrs = rrs
        self.b443 = b443
        self.b555 = b555
        self.b670 = b670
        self.b490 = 2
        self.b565 = 4
        self.b520 = 3

    def value(self):
        chl_1 = self.chl_ci()
        chl_2 = self.chl_oc4()
        x = (chl_1 - 0.15) / (0.20 - 0.15)
        y = (0.20 - chl_1) / (0.20 - 0.15)
        chl_3 = x * chl_2 + y * chl_1
        chl_1[chl_1 > 0.15] = np.nan
        chl_2[chl_1 < 0.2] = np.nan
        chl_3[(chl_1 <= 0.15) | (chl_1 >= 0.2)] = np.nan
        chl_value = np.nanmean(np.dstack([chl_1, chl_2, chl_3]), axis=2)
        return chl_value

    def chl_ci(self):
        c = np.array([-0.4909, 191.6590])
        rrs1 = self.rrs[:, :, self.b443]
        rrs2 = self.rrs[:, :, self.b555]
        rrs3 = self.rrs[:, :, self.b670]
        ci = rrs2 - 0.5 * rrs1 + rrs3
        chl = np.power(10.0, c[0] + c[1] * ci)
        return chl

    def chl_oc4(self):
        from l2gen import get_chl

        a = [0.3325, -2.8278, 3.0939, -2.0917, -0.0257]  # h1ccocts
        chl2 = get_chl.chl_oc4(rrs=self.rrs, b443=self.b443, b490=self.b490, b520=self.b520, b565=self.b565, a=a)
        return chl2
