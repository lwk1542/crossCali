# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 21:04
@FileName: launch.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
SDG-1 MII定标,模拟TOA radiance
"""
import os

from utils import cross_calibration_sdgsat1miiV2
from sharepy import spatial_limit


def ref_bands(sensorID_ref):
    if sensorID_ref == "s3b_olci":
        sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
    elif sensorID_ref == "s3a_olci":
        sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
    else:
        sensorID_ref_bands = [0]
    return sensorID_ref_bands


if __name__ == '__main__':
    from sensor import sdgsat1mii

    father_dir = r"G:\SDGsat\calibration\sea\202303supply\new"
    target_path = father_dir + os.sep + "target"
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
        print("cross calibration...>>>")
        limit = spa_lim[subdir]
        rangelonlat = [limit[0], limit[3], limit[2], limit[1]]
        roifile = sdgsat1mii.roi(filepath=filedir, rangelonlat=rangelonlat, resize=0.03)  # 小于1，分辨率降低
        target_file = roifile  # filedir+os.sep+subdir+"_ROI.tif"
        sensorID_ref = "s3a_olci"
        sensorID_ref_bands = ref_bands(sensorID_ref)

        simulate = cross_calibration_sdgsat1miiV2.SimulationLtoa(target_file=target_file,
                                                                 farther_dir=father_dir,
                                                                 sensorid_ref=sensorID_ref,
                                                                 ref_bands=sensorID_ref_bands)
        toafile = simulate.run_main()
        if toafile.__len__() == 0:
            print("No generating cross calibration file, the received returned value is {}:".format(toafile))
            continue
