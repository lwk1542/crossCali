# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/12 15:34
@FileName: mask.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np


def mask(sza=None, saa=None, vza=None, vaa=None, sza_ref=None, saa_ref=None, vza_ref=None, vaa_ref=None):
    """
    对散射角、观测几何做限定
    """
    reaa = saa - vaa
    reaa = np.abs(reaa)
    reaa[reaa > 180.] = reaa[reaa > 180.] - 180
    temp = np.sqrt((1. - np.cos(vza * np.pi / 180.) ** 2) * (1. - np.cos(sza * np.pi / 180.) ** 2)) * np.cos(
        reaa * np.pi / 180.)

    temp_1 = -np.cos(vza * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_1[temp_1 < -1.] = -1.
    scatt1 = np.arccos(temp_1) * 180 / np.pi
    temp_2 = np.cos(vza * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_2[temp_2 > 1.] = 1.
    scatt2 = np.arccos(temp_2) * 180 / np.pi

    reaa_ref = saa_ref - vaa_ref
    reaa_ref = np.abs(reaa_ref)
    reaa_ref[reaa_ref > 180.] = reaa_ref[reaa_ref > 180.] - 180
    temp = np.sqrt((1. - np.cos(vza_ref * np.pi / 180.) ** 2) * (1. - np.cos(sza * np.pi / 180.) ** 2)) * np.cos(
        reaa_ref * np.pi / 180.)

    temp_1 = -np.cos(vza_ref * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_1[temp_1 < -1.] = -1.
    scatt1_ref = np.arccos(temp_1) * 180 / np.pi
    temp_2 = np.cos(vza_ref * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_2[temp_2 > 1.] = 1.
    scatt2_ref = np.arccos(temp_2) * 180 / np.pi

    mask1 = scatt1 - scatt1_ref
    mask1[np.isnan(mask1)] = 999
    mask1[mask1 > 5] = np.nan
    mask1 = mask1 / mask1

    mask2 = scatt2 - scatt2_ref
    mask2[np.isnan(mask2)] = 999
    mask2[mask2 > 5] = np.nan
    mask2 = mask2 / mask2

    sza_mask3 = sza - sza_ref
    sza_mask3[np.isnan(sza_mask3)] = 999
    sza_mask3[sza_mask3 > 5] = np.nan
    sza_mask3 = sza_mask3 / sza_mask3

    sza[np.isnan(sza)] = 999
    sza[sza > 30] = np.nan
    sza_mask1 = sza / sza

    vza[np.isnan(vza)] = 999
    vza[vza > 30] = np.nan
    sza_mask2 = vza / vza
    mask = mask1 * mask2 * sza_mask1 * sza_mask2 * sza_mask3

    return mask


def mask_2(sza: np.ndarray, saa: np.ndarray, vza: np.ndarray, vaa: np.ndarray,
           sza_ref: np.ndarray, saa_ref: np.ndarray, vza_ref: np.ndarray, vaa_ref: np.ndarray):
    """
    Chen, Jun, Xianqiang He, Zhongli Liu, Na Xu, Lingling Ma, Qianguo Xing, Xiuqing Hu, and Delu Pan.
    "An approach to cross-calibrating multi-mission satellite data for the open ocean." Remote Sensing of Environment
    246 (2020): 111895.
    :param sza:
    :param saa:
    :param vza:
    :param vaa:
    :param sza_ref:
    :param saa_ref:
    :param vza_ref:
    :param vaa_ref:
    :return:
    """
    sza[sza > 50] = np.nan
    sza_ref[sza > 50] = np.nan
    mask1 = sza_ref/sza_ref

    vza[vza > 38] = np.nan
    mask2 = vza/vza

    vza_ref[vza_ref > 38] = np.nan
    mask3 = vza_ref/vza_ref

    reaa = saa - vaa
    reaa = np.abs(reaa)
    reaa[reaa > 180.] = reaa[reaa > 180.] - 180
    reaa[abs(reaa) > 150] = np.nan
    mask4 = reaa/reaa

    reaa_ref = saa_ref - vaa_ref
    reaa_ref = np.abs(reaa_ref)
    reaa_ref[reaa_ref > 180.] = reaa_ref[reaa_ref > 180.] - 180
    reaa_ref[reaa_ref > 150] = np.nan
    mask5 = reaa_ref / reaa_ref

    return mask1*mask2*mask3*mask4*mask5
