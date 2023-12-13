# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/12 10:14
@FileName: resize.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from scipy import interpolate
import skimage.measure
import numpy as np


def down_sample(sza, saa, vza, vaa, lon, lat, resize):
    sza_upscale = skimage.measure.block_reduce(sza, block_size=(resize, resize),
                                               func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    vza_upscale = skimage.measure.block_reduce(vza, block_size=(resize, resize),
                                               func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    saa_upscale = skimage.measure.block_reduce(saa, block_size=(resize, resize),
                                               func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    vaa_upscale = skimage.measure.block_reduce(vaa, block_size=(resize, resize),
                                               func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    lon_upscale = skimage.measure.block_reduce(lon, block_size=(resize, resize),
                                               func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    lat_upscale = skimage.measure.block_reduce(lat, block_size=(resize, resize),
                                               func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    return sza_upscale, saa_upscale, vza_upscale, vaa_upscale, lat_upscale, lon_upscale


def down_sample_sentinel(data_list, resize):
    [rh, pressure, water_vapor, o3, wind_speed, winddirection] = data_list
    # temper_upscale = skimage.measure.block_reduce(temper, block_size=(resize, resize),
    #                                               func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    rh_upscale = skimage.measure.block_reduce(rh, block_size=(resize, resize),
                                              func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    pressure_upscale = skimage.measure.block_reduce(pressure, block_size=(resize, resize),
                                                    func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    water_vapor_upscale = skimage.measure.block_reduce(water_vapor, block_size=(resize, resize),
                                                       func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    o3_upscale = skimage.measure.block_reduce(o3, block_size=(resize, resize),
                                              func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    wind_speed_upscale = skimage.measure.block_reduce(wind_speed, block_size=(resize, resize),
                                                      func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    winddirection_upscale = skimage.measure.block_reduce(winddirection, block_size=(resize, resize),
                                                         func=np.nanmean, cval=np.nan,
                                                         func_kwargs={'dtype': np.float32})
    return [rh_upscale, pressure_upscale, water_vapor_upscale, o3_upscale, wind_speed_upscale, winddirection_upscale]


def down_sample_aerosol(lt,  resize, band_nirs, band_nirl):

    Lt_upscale = skimage.measure.block_reduce(lt, block_size=(resize, resize, 1),
                                              func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    Lt_nirs = skimage.measure.block_reduce(lt[:, :, band_nirs], block_size=(resize, resize),
                                           func=np.nanmin, cval=np.nan)
    Lt_nirl = skimage.measure.block_reduce(lt[:, :, band_nirs], block_size=(resize, resize),
                                           func=np.nanmin, cval=np.nan)
    Lt_upscale[:, :, band_nirs] = Lt_nirs
    Lt_upscale[:, :, band_nirl] = Lt_nirl
    # t_sol = skimage.measure.block_reduce(t_sol, block_size=(resize, resize, 1),
    #                                      func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    # t_sen = skimage.measure.block_reduce(t_sen, block_size=(resize, resize, 1),
    #                                      func=np.nanmean, cval=np.nan, func_kwargs={'dtype': np.float32})
    return Lt_upscale


def up_sample(data: np.ndarray = None, lat: np.ndarray = None, lon: np.ndarray = None, lat_tar: np.ndarray = None,
              lon_tar: np.ndarray = None):
    data_tar = np.zeros(shape=(lat_tar.shape[0], lat_tar.shape[1], data.shape[2]))
    for band_num in range(data.shape[2]):
        data_tar[:, :, band_num] = interpolate.griddata((lat.flatten(), lon.flatten()), data[:, :, band_num].flatten(),
                                                        (lat_tar, lon_tar), method='nearest')
    return data_tar
