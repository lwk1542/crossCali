# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/14 21:26
@FileName: extra_info.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from . import getfile


def extra_info(tar_filedir, date_str2):
    filepath = getfile(tar_filedir, date_str2)
    if filepath is None:
        print("#####没有目标文件################################################")
    lut_path = r'share/sdgsat1mii'
    # 根据卫星传感器指定两个用于气溶胶估算的近红外波段，起始位0
    nirs_num = 5
    nirl_num = 6
    return filepath, lut_path, nirs_num, nirl_num
