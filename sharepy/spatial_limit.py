# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/14 22:26
@FileName: spatial_limit.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
一个txt文件，里面存储了文件名和对应的要裁剪的范围:
KX10_MII_20230315_E115.41_N10.65_202300092684_L4A,114.5,9.5,115.4,10.2
KX10_MII_20230320_E114.17_N7.43_202300122491_L4A,113,6.7,114,7.8
KX10_MII_20230320_E114.71_N10.15_202300122492_L4A,113.86,11,114.26,11.5
KX10_MII_20230325_E113.10_N8.68_202300160907_L4A,111.77,9.07,112.5,9.9
KX10_MII_20230325_E113.64_N11.41_202300160905_L4A,112.93,9.85,113.5,10.71
"""
import os


def spatial_limit(txtfile: str) -> dict:
    dic = {}
    with open(txtfile, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            a = line.split(",")
            dic[a[0]] = [float(a[1]), float(a[2]), float(a[3]), float(a[4])]
            print(line)
    return dic


if __name__ == '__main__':
    file = "G:\SDGsat\calibration\sea" + os.sep + "spatialLimit.txt"
    spatial_limit(txtfile=file)
