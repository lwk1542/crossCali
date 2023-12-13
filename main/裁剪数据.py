# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/5 15:09
@FileName: 裁剪数据.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""


def sdgsat1mii():
    import os
    from sharepy import spatial_limit
    from sensor import sdgsat1mii
    target_path = r"G:\SDGsat\calibration\sea\2023\calibration\test"
    subdirs = os.listdir(target_path)
    print("total {} images:".format(subdirs.__len__()), subdirs)
    spatialScopefile = r"G:\SDGsat\calibration\sea\2023/spatialLimit.txt"
    spa_lim = spatial_limit(txtfile=spatialScopefile)

    for i, subdir in enumerate(subdirs):
        filedir = target_path + os.sep + subdir
        if not os.path.isdir(filedir):
            continue
        if ("KX10_MII_" not in subdir) or ("L4B" not in subdir):
            continue
        print("================处理第{}个目标文件:{}:开始=================".format(i, subdir))
        limit = spa_lim[subdir]
        rangelonlat = [limit[0], limit[3], limit[2], limit[1]]
        roifile = sdgsat1mii.roi(filepath=filedir, rangelonlat=rangelonlat, resize=0.03)  # 小于1，分辨率降低

    return


if __name__ == '__main__':
    sdgsat1mii()