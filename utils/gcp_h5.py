# -*- coding: utf-8 -*-
"""
@Time    : 2023/3/16 16:35
@FileName: gcp_h5.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
python3.7(base1)
"""
# gcp校正:
from osgeo import osr
from osgeo import gdal
import numpy as np
import h5py
import glob
import os
import cv2


def erode(data):
    # "图像腐蚀"
    ret, binary = cv2.threshold(data, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5))
    open1 = cv2.erode(binary, kernel, iterations=5)  # 腐蚀
    open1[np.isnan(open1)] = np.nan
    open1[open1 < 0] = np.nan
    open1[open1 > 260] = np.nan
    open1[~np.isnan(open1)] = 1
    data_ = data * open1
    return data_

# 创建临时文件
def write_bands(im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    # 数据类型必须有，因为要计算需要多大内存空间
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", im_width, im_height, im_bands, datatype)

    # 写入数组数据
    if im_bands == 1:
        # dataset.GetRasterBand(1).SetNoDataValue(65535)
        try:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入
        except:
            dataset.GetRasterBand(1).WriteArray(im_data[:, :, 0])
    else:
        # if banddes==None:
        # banddes = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_520', 'Rrs_565', 'Rrs_670', 'chlor_a']
        for i in range(im_bands):
            try:
                # dataset.GetRasterBand(i + 1).SetNoDataValue(65535)
                RasterBand = dataset.GetRasterBand(i + 1)
                # RasterBand.SetDescription(banddes[i])
                RasterBand.WriteArray(im_data[:, :, i])
            except IndentationError:
                print('band:' + i)

    return dataset


class Sdg(object):
    def __init__(self):
        pass

    def run_main(self):
        self.run_cross_calibration()

    def run_cross_calibration(self):
        files_path = r"G:\SDGsat\calibration\sea\202303\result"
        files= glob.glob(files_path+os.sep+"KX10_MII_*crossCalibration.h5")
        for file in files:
            outfile = os.path.dirname(file)+os.sep+os.path.splitext(os.path.basename(file))[0]+"_geo.tif"
            self.cross_calibration_file(infile=file, outfile=outfile)

    def cross_calibration_file(self, infile: str, outfile: str=None, radi_bands=["401.0", "438.0", "495.0", "553.0","657.0", "776.0", "854.0"]):
        if not outfile:
            outfile = os.path.dirname(infile) + os.sep + os.path.splitext(os.path.basename(infile))[0] + "_geo.tif"
        f = h5py.File(infile, "r")
        longitude = f['Navigation Data/lon'][()]
        latitude = f['Navigation Data/lat'][()]
        gains = np.array(
            [0.051560133, 0.036241353, 0.023316835, 0.015849666, 0.016096381, 0.019719039, 0.013811458]) * 0.1
        # bands = ['Rrs412', 'Rrs443', 'Rrs490', 'Rrs520', 'Rrs565', 'Rrs670', 'chlor_a']
        value = np.empty((longitude.shape[0], longitude.shape[1], radi_bands.__len__()*2))
        for i, band in enumerate(radi_bands):
            value_band = f['Geophysical Data/DN_' + radi_bands[i]][()]*gains[i]
            value_band[value_band == -32767.] = np.nan
            value_band[value_band < -32767.] = np.nan
            value_band[value_band > 32767.] = np.nan
            value_band[value_band <= 0] = np.nan
            value[:, :, i] = value_band

            value_band = f['Geophysical Data/Lt_simu_' + radi_bands[i]][()]
            value_band[value_band == -32767.] = np.nan
            value_band[value_band < -32767.] = np.nan
            value_band[value_band > 32767.] = np.nan
            value_band[value_band <= 0] = np.nan
            value[:, :, i+7] = value_band


        # judge1 = value[:, :, 4] < value[:, :, 6]
        # value[judge1] = np.nan
        # judge2 = ((value[:, :, 0] >0.08)| (value[:, :, 1]>0.08))&((value[:, :, 2] >0.08)| (value[:, :, 3]>0.08))&(value[:, :, 6]>0.03)
        # value[judge2] = np.nan
        # ret, binary = cv2.threshold(value[:, :, 0], 0, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((5, 5))
        # open1 = cv2.erode(binary, kernel, iterations=3)         # 腐蚀
        # # open1 = cv2.dilate(binary, kernel)  # 膨胀图像
        # # open1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)  # 开运算
        # # open1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # 开运算

        # open1[np.isnan(open1)] = np.nan
        # open1[open1 < 0] = np.nan
        # open1[open1 > 260] = np.nan
        # open1=open1/open1
        # value = value * open1.reshape(open1.shape[0], open1.shape[1], 1)
        # value[np.isnan(value)] = 0
        # value = (value * 10000).astype(int)
        f.close()

        # 将波段数据写入临时内存文件
        image: gdal.Dataset = write_bands(value)

        # 控制点列表, 设置7*7个控制点
        gcps = []
        x_arr = np.linspace(0, longitude.shape[1] - 1, num=7, endpoint=True, dtype=int)
        y_arr = np.linspace(0, longitude.shape[0] - 1, num=7, endpoint=True, dtype=int)
        for x in x_arr:
            for y in y_arr:
                if abs(longitude[y, x]) > 180 or abs(latitude[y, x]) > 90:
                    continue
                gcps.append(gdal.GCP(np.float64(longitude[y, x]), np.float64(latitude[y, x]),
                                     0,
                                     np.float64(x), np.float64(y)))

        # 设置空间参考
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        sr.MorphToESRI()

        # 给数据及设置控制点及空间参考
        image.SetGCPs(gcps, sr.ExportToWkt())

        cutlinelayer = radi_bands
        dst = gdal.Warp(outfile, image, format='GTiff', tps=True, xRes=0.001, yRes=0.001, dstNodata=np.nan,
                        resampleAlg=gdal.GRA_NearestNeighbour)  # dstNodata=65535

        for i, bandname in enumerate(cutlinelayer):
            band = dst.GetRasterBand(i + 1)
            band.SetMetadata({'bandname': bandname})
            band.SetDescription(bandname)

        image: None
        return outfile


if __name__ == '__main__':
    Sdg().run_main()
