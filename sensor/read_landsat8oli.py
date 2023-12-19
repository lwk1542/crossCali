# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: read_landsat8oli.py
@time: 2022/1/9 21:37
@desc:
"""
import os
import numpy as np
import readtif
from sensor.landsat8oli import landsat_metadata
import tarfile




def oli_info(infile):
    archive = tarfile.open(infile)
    outpath = infile[infile.rfind("/") + 1:infile.find(".")]
    print("解压缩...")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for name in archive.getnames():
        if not os.path.exists(os.path.join(outpath, name)):
            archive.extract(name, path=outpath)
    mtl_file=outpath+os.sep+os.path.basename(outpath)+"_MTL.txt"
    meta = landsat_metadata.landsat_metadata(mtl_file)
    date=meta.DATE_ACQUIRED
    year, month, day = int(date[0:4]), int(date[5:7]), int(date[8:])
    hour, minute, second = 10, 15, 0
    saa_=meta.SUN_AZIMUTH
    sza_=90-meta.SUN_ELEVATION
    band_list = ["B1.TIF", "B2.TIF", "B3.TIF", "B4.TIF", "B5.TIF", "B6.TIF", "B7.TIF"]

    # 读完一个tif后，MTL.txt文件会被莫名其妙的删除掉，不知道原因；在处理tif前，只能先把相应的参数全部读出来

    for i, band in enumerate(band_list):
        band_path = mtl_file.replace("MTL.txt", band)
        # print(band_path)
        [lon_grid, lat_grid, band_value] = readtif.run(band_path)
        band_value[band_value == 0] = np.nan
        if i == 0:
            data = np.empty(shape=(band_value.shape[0], band_value.shape[1], band_list.__len__()))
        ml = getattr(meta, "RADIANCE_MULT_BAND_{0}".format(i + 1))  # multiplicative scaling factor
        al = getattr(meta, "RADIANCE_ADD_BAND_{0}".format(i + 1))  # additive rescaling factor

        data[:, :, i] = (band_value * ml + al)  # 注意单位的变化

    sza = np.full_like(band_value, fill_value=sza_)
    saa = np.full_like(band_value, fill_value=saa_)
    vza, vaa = np.full_like(lon_grid, fill_value=0.), np.full_like(lon_grid, fill_value=0.5)
    num_443, num_490, num_520, num_555, num_670, nirs_num, nirl_num = 0, 1, None, 2, 3, 4, 5,
    nwvis = 4
    red = num_670
    return [sza, vza, saa, vaa, lat_grid, lon_grid, data, year, month, day, num_443, num_490, num_520, num_555, num_670,
            nirs_num, nirl_num, nwvis, red]


# test
if __name__ == '__main__':
    # oli_info(r"F:\cali_spatial_vari\OLI\LC08_L1TP_064045_20200126_20200210_01_T1")
    # l1_metadata(r"F:\cali_spatial_vari\OLI\LC08_L1TP_064045_20200126_20200210_01_T1")
    infile=r"F:\cali_spatial_vari\OLI/LC08_L1TP_064045_20200126_20200210_01_T1.tar.gz"
    archive = tarfile.open(infile)
    LSname = infile[infile.rfind("/") + 1:infile.find(".")]
    outpath=os.path.dirname(infile)+os.sep+LSname
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for name in archive.getnames():
        if not os.path.exists(os.path.join(outpath, name)):
            archive.extract(name, path=outpath)
    print()