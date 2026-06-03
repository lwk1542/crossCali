# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/4 16:04
@FileName: read_aerosol_seadas.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
交叉定标需要用到seadas处理的参考影像，通过这个文件读取
"""
from pathlib import Path
import logging

import numpy as np
import h5py
import os

from sharepy import read_sensorinfo

class info(object):
    def __init__(self, file: str):
        self.file = file
        pass

    def run(self):
        fileid = os.path.basename(self.file)
        if "S3A" in fileid:
            self.bands, self.F0, self.nirl = sensor(sensor_id="s3a_olci")
        if "S3B" in fileid:
            self.bands, self.F0, self.nirl = sensor(sensor_id="s3b_olci")
        if "MOD" in fileid:
            self.bands, self.F0, self.nirl = sensor(sensor_id="terra_modis")
        if "MYD" in fileid:
            self.bands, self.F0, self.nirl = sensor(sensor_id="aqua_modis")

        value = self.read_aerosol_info_seadas()
        return value

    def read_aerosol_info_seadas(self):
        # seadas处理出来的文件
        parameters = {}
        f = h5py.File(self.file, mode='r')
        aer_model_max = f["/geophysical_data/aer_model_max"][()] * 1. - 1
        aer_model_max[(aer_model_max < 0) | (aer_model_max > 79)] = np.nan
        parameters['aer_model_max'] = aer_model_max
        aer_model_min = f["/geophysical_data/aer_model_min"][()] * 1. - 1
        aer_model_min[(aer_model_min < 0) | (aer_model_min > 79)] = np.nan
        parameters['aer_model_min'] = aer_model_min
        aer_model_ratio = f["/geophysical_data/aer_model_ratio"][()]
        aer_model_ratio[(aer_model_ratio < 0) | (aer_model_ratio > 1)] = np.nan
        parameters['aer_model_ratio'] = aer_model_ratio
        La_nirl = f["/geophysical_data/La_" + str(self.nirl)][()] / 10.
        La_nirl[(La_nirl < 0) | (La_nirl > 10)] = np.nan
        parameters['La_nirl'] = La_nirl

        p = self.bands.index(self.nirl)
        F0_nirl = self.F0[p]
        parameters['F0_nirl'] = F0_nirl
        bands = self.bands[0:p + 1]
        F0 = self.F0[0:p + 1]

        Rrs = np.full(shape=(La_nirl.shape[0], La_nirl.shape[1], bands.__len__()), fill_value=np.nan)
        for i, band in enumerate(bands):
            vari = f["/geophysical_data/Rrs_" + str(int(band))]
            offset = vari.attrs['add_offset'][0]
            scale = vari.attrs['scale_factor'][0]
            valid_max = vari.attrs['valid_max'][0]
            valid_min = vari.attrs['valid_min'][0]
            value = vari[()] * 1.
            value[(value < valid_min) | (value > valid_max)] = np.nan
            Rrs[:, :, i] = value * scale + offset
        parameters['Rrs'] = Rrs
        # for i, band in enumerate(bands):
        #     taua[:,:,i]=f["/aerosol_information/aot_"+str(band)][()]

        lat = f["/navigation_data/latitude"][()]
        lat[(lat < -90) | (lat > 90)] = np.nan
        lon = f["/navigation_data/longitude"][()]
        lon[(lon < -180) | (lon > 180)] = np.nan
        parameters['latitude'] = lat
        parameters['longitude'] = lon

        metro = ['solz', 'sola', 'senz', 'sena', 'windangle','windspeed',  'pressure', 'water_vapor', 'ozone',
                 'humidity', 'no2_strat', 'no2_tropo']# 'windangle',

        for para_index in metro:
            vari = f[f"/geophysical_data/{para_index}"]
            try:
                offset = vari.attrs['add_offset'][0]
            except:
                offset = 0
            try:
                scale = vari.attrs['scale_factor'][0]
            except:
                scale = 1
            try:
                valid_max = vari.attrs['valid_max'][0]
            except:
                valid_max = np.inf
            try:
                valid_min = vari.attrs['valid_min'][0]
            except:
                valid_min = -np.inf
            value = vari[()] * 1.
            value[(value < valid_min) | (value > valid_max)] = np.nan
            value = value * scale + offset
            parameters[para_index] = value
        f.close()

        return parameters

def sensor(sensor_id: str):
    """
    根据传感器 ID 获取波段、F0 和近红外参考波段 (nirl)
    """
    # 1. 动态定位项目根目录下的 share 文件夹
    base_share = Path(__file__).resolve().parent.parent / "share"
    
    # 2. 定义传感器配置文件映射 (传感器ID: (相对路径, nirl_wavelength))
    # 这样以后增加新传感器只需要在这里加一行，不需要写新函数
    sensor_config = {
        "s3a_olci":    ("olci/s3a/msl12_sensor_info.dat", 865),
        "s3b_olci":    ("olci/s3b/msl12_sensor_info.dat", 865),
        "terra_modis": ("modis/terra/msl12_sensor_info.dat", 869),
        "aqua_modis":  ("modis/aqua/msl12_sensor_info.dat", 869)
    }

    # 3. 获取配置信息
    config = sensor_config.get(sensor_id.lower())
    if not config:
        raise ValueError(f"未定义的传感器 ID: {sensor_id}")

    rel_path, nirl = config
    sensorinfo_file = base_share / rel_path

    # 4. 路径存在性检查
    if not sensorinfo_file.exists():
        logging.error(f"传感器配置文件缺失: {sensorinfo_file}")
        raise FileNotFoundError(f"Missing: {sensorinfo_file}")

    # 5. 调用读取函数并解包
    # 假设 read_sensorinfo 接受 Path 对象或字符串
    try:
        sensor_info = read_sensorinfo(sensorinfo_file=str(sensorinfo_file))
        # 建议只提取你需要的变量，如果 sensor_info 返回值很多，解包要小心
        bands = sensor_info[0]
        F0 = sensor_info[1]
        
        logging.info(f"成功加载 {sensor_id} 参数，nirl: {nirl}")
        return bands, F0, nirl
        
    except Exception as e:
        logging.error(f"解析 {sensor_id} 配置文件失败: {e}")
        raise




# def sensor(sensor: str):


#     def s3a_olci():
#         nirl = 865
#         sensor_info = read_sensorinfo(sensorinfo_file="../share/olci/s3a/msl12_sensor_info.dat")
#         [bands, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
#          oobwv, ooblw, wed, waph] = sensor_info
#         return bands, F0, nirl

#     def s3b_olci():
#         nirl = 865
#         sensor_info = read_sensorinfo(sensorinfo_file="../share/olci/s3b/msl12_sensor_info.dat")
#         [bands, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
#          oobwv, ooblw, wed, waph] = sensor_info
#         return bands, F0, nirl

#     def terra_modis():
#         nirl = 869
#         sensor_info = read_sensorinfo(sensorinfo_file="../share/modis/hmodist/msl12_sensor_info.dat")
#         [bands, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
#          oobwv, ooblw, wed, waph] = sensor_info
#         return bands, F0, nirl
#     def aqua_modis():
#         nirl = 869
#         sensor_info = read_sensorinfo(sensorinfo_file="../share/modis/hmodisa/msl12_sensor_info.dat")
#         [bands, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
#          oobwv, ooblw, wed, waph] = sensor_info
#         return bands, F0, nirl

#     if sensor == "s3a_olci":
#         return s3a_olci()
#     elif sensor == "s3b_olci":
#         return s3b_olci()
#     elif sensor == "terra_modis":
#         return terra_modis()
