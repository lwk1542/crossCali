# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/13 10:12
@FileName: crossCalib_sdgsat1miiV2.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import os

from utils import cross_calibration_sdgsat1miiv3
from sharepy import spatial_limit


def ref_bands(sensorID_ref):
    if sensorID_ref == "s3b_olci":
        sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
    elif sensorID_ref == "s3a_olci":
        sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
    else:
        sensorID_ref_bands = [0]
    return sensorID_ref_bands


def matchup(txtfile: str) -> dict:
    obj = open(txtfile, "r")
    lines = obj.readlines()
    dic = {}
    for line in lines:
        _ = line.split(";")
        dic[_[0]] = [i for i in _[1:]]
    return dic


if __name__ == '__main__':
    from sensor import sdgsat1mii

    father_dir = r"G:\SDGsat\calibration\sea\2023\supply\case1"
    target_path = father_dir + os.sep + "target"
    subdirs = os.listdir(target_path)
    print("total {} images:".format(subdirs.__len__()), subdirs)
    Scopefile = r"G:\SDGsat\calibration\sea\2023\supply\case1/download_log.txt"
    match_dic = matchup(txtfile=Scopefile)
    for i, key in enumerate(match_dic.keys()):
        filedir = target_path + os.sep + key
        if not os.path.isdir(filedir):
            continue
        if ("KX10_MII_" not in key) or ("L4B" not in key):
            continue
        print("================处理第{}个目标文件:{}:开始=================".format(i, key))
        print("cross calibration...>>>")
        target_file = filedir + os.sep + key + "_ROI.tif"
        for j, _ in enumerate(match_dic[key]):
            if _.__len__() < 5:
                continue
            id_ = ".".join(_.split(".")[:-1]) + ".SEN3_seadas_rrs.hdf"
            print("================******第{}个参考文件******==============".format(j, id_))
            reference_file = father_dir + os.sep + "reference" + os.sep + id_
            simulate = cross_calibration_sdgsat1miiv3.SimulationLtoa(target_file=target_file,
                                                                     farther_dir=father_dir,
                                                                     ref_file=reference_file)
            toafile = simulate.run_main()
            if toafile == 0:
                continue
            if toafile == 1:
                continue
