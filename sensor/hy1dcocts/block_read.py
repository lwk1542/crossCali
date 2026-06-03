# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/8 16:17
@FileName: block_read.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import h5py
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class ReadIterator(object):
    """
    2022.11.7
    这是一个迭代器，通过块状读取、块状处理，利于节约内存开支
    """

    def __init__(self, in_file, blocksize: int = None, rows: int = None, columns: int = None):
        """
        Args:
            in_file: 文件路径
            blocksize: 一次要处理的数据的行数，默认50
        """
        self.in_file = in_file  # 文件
        self.blocksize = blocksize
        self.XSize = columns  # 网格的X轴像素数量
        self.YSize = rows  # 网格的Y轴像素数量
        self.bands_name = [412, 443, 490, 520, 565, 670, 750, 865]
        self.iter_count = self.YSize // self.blocksize
        self.surplus_rows = self.YSize % self.blocksize
        if self.surplus_rows == 0:
            self.iter_count -= 1
        print("分块数量={0}".format(self.iter_count))
        print("imagery info:rows={0},columns={1},bands={2},block_rows={3},"
              "surplus_rows={4}".format(self.YSize, self.XSize, self.bands_name.__len__(), self.blocksize,
                                        self.surplus_rows))
        self.iter_num = 0
        self.lag = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_num <= self.iter_count:
            if (self.iter_num == self.iter_count) and (self.surplus_rows > 0):
                y_offset = self.surplus_rows
            else:
                y_offset = self.blocksize
            size = self.bands_name.__len__()
            data = np.zeros(shape=(y_offset, self.XSize, size)) + np.nan
            ds = h5py.File(self.in_file, mode="r")
            for i, band_name in enumerate(self.bands_name):
                try:
                    data[:, :, i] = ds["Geophysical Data/DN_"+str(band_name)][self.lag:self.lag+y_offset, :]
                except:
                    data[:, :, i] = ds["Geophysical Data/L_"+str(band_name)][self.lag:self.lag+y_offset, :]

            navi_ = ["Latitude", "Longitude", "Sun Zenith Angle", "Sun Azimuth Angle", "Satellite Zenith Angle",
                     "Satellite Azimuth Angle"]
            latitude = ds["Navigation Data/" + "Latitude"][self.lag:self.lag + y_offset, :]
            longitude = ds["Navigation Data/" + "Longitude"][self.lag:self.lag + y_offset, :]
            sza = ds["Navigation Data/" + "Sun Zenith Angle"][self.lag:self.lag + y_offset, :]
            saa = ds["Navigation Data/" + "Sun Azimuth Angle"][self.lag:self.lag + y_offset, :]
            vza = ds["Navigation Data/" + "Satellite Zenith Angle"][self.lag:self.lag + y_offset, :]
            vaa = ds["Navigation Data/" + "Satellite Azimuth Angle"][self.lag:self.lag + y_offset, :]

            cal = ds["Calibration/" + "Vicarious Calibration gain factor"][()]
            gains = cal.reshape(-1)[0:8]
            
            offsets = np.zeros_like(gains)
            gains = np.zeros_like(gains)+1.
            # offsets = ds["Calibration/" + "Calibration Coefficients Offsets factor"][0:8]
            # gains = ds["Calibration/" + "Calibration Coefficients Scale factor"][0:8]
            
            # gains = cal.reshape(-1)[0:8]
            # offsets = np.zeros_like(gains)
            # gains = np.zeros_like(gains)+1.

            self.iter_num += 1
            self.lag = self.iter_num * y_offset
            data = np.rot90(data, 2)
            longitude = np.rot90(longitude, 2)
            latitude = np.rot90(latitude, 2)
            vaa = np.rot90(vaa, 2)
            vza = np.rot90(vza, 2)
            saa = np.rot90(saa, 2)
            sza = np.rot90(sza, 2)
            return data, gains, offsets, longitude, latitude, vaa, vza, saa, sza
        else:
            raise StopIteration()  # 表示至此停止迭代


if __name__ == '__main__':
    infiles = [
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_A.tif",
        r"E:\SDGSAT-1\test\KX10_MII_20220407_E115.85_N23.36_202200035633_L4A/KX10_MII_20220407_E115.85_N23.36_202200035633_L4A_B.tif"]
    for i in ReadIterator(infiles[0]):
        data, lon, lat = i
        print(data.shape)
