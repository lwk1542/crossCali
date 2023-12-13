# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/4 17:28
@FileName: resample.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np
import skimage.measure


def resampe(data: np.array(float | int), block_size: int, dtype: np.dtype) -> np.array():
    return skimage.measure.block_reduce(data, block_size=(block_size, block_size, 1),
                                        func=np.nanmean, cval=np.nan, func_kwargs={'dtype': dtype})
