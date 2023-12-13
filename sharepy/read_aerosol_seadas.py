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
            self.bands, self.F0, self.nirl = sensor(sensor="s3a_olci")
        if "S3B" in fileid:
            self.bands, self.F0, self.nirl = sensor(sensor="s3b_olci")
        if "MOD" in fileid:
            self.bands, self.F0, self.nirl = sensor(sensor="terra_modis")

        value = self.read_aerosol_info_seadas()
        return value

    def read_aerosol_info_seadas(self):
        # seadas处理出来的文件
        f = h5py.File(self.file, mode='r')
        aer_model_max = f["/geophysical_data/aer_model_max"][()] * 1. - 1
        aer_model_max[(aer_model_max < 0) | (aer_model_max > 79)] = np.nan
        aer_model_min = f["/geophysical_data/aer_model_min"][()] * 1. - 1
        aer_model_min[(aer_model_min < 0) | (aer_model_min > 79)] = np.nan
        aer_model_ratio = f["/geophysical_data/aer_model_ratio"][()]
        aer_model_ratio[(aer_model_ratio < 0) | (aer_model_ratio > 1)] = np.nan
        La_nirl = f["/geophysical_data/La_" + str(self.nirl)][()]/10.
        La_nirl[(La_nirl < 0) | (La_nirl > 10)] = np.nan

        p = self.bands.index(self.nirl)
        F0_nirl = self.F0[p]
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

        # for i, band in enumerate(bands):
        #     taua[:,:,i]=f["/aerosol_information/aot_"+str(band)][()]

        lat = f["/navigation_data/latitude"][()]
        lat[(lat < -90) | (lat > 90)] = np.nan
        lon = f["/navigation_data/longitude"][()]
        lon[(lon < -180) | (lon > 180)] = np.nan

        vari = f["/geophysical_data/solz"]
        offset = vari.attrs['add_offset'][0]
        scale = vari.attrs['scale_factor'][0]
        valid_max = vari.attrs['valid_max'][0]
        valid_min = vari.attrs['valid_min'][0]
        value = vari[()] * 1.
        value[(value < valid_min) | (value > valid_max)] = np.nan
        sza = value * scale + offset
        del vari, value
        vari = f["/geophysical_data/senz"]
        offset = vari.attrs['add_offset'][0]
        scale = vari.attrs['scale_factor'][0]
        valid_max = vari.attrs['valid_max'][0]
        valid_min = vari.attrs['valid_min'][0]
        value = vari[()] * 1.
        value[(value < valid_min) | (value > valid_max)] = np.nan
        vza = value * scale + offset
        del vari, value
        vari = f["/geophysical_data/sola"]
        offset = vari.attrs['add_offset'][0]
        scale = vari.attrs['scale_factor'][0]
        valid_max = vari.attrs['valid_max'][0]
        valid_min = vari.attrs['valid_min'][0]
        value = vari[()] * 1.
        value[(value < valid_min) | (value > valid_max)] = np.nan
        saa = value * scale + offset
        del vari, value
        vari = f["/geophysical_data/sena"]
        offset = vari.attrs['add_offset'][0]
        scale = vari.attrs['scale_factor'][0]
        valid_max = vari.attrs['valid_max'][0]
        valid_min = vari.attrs['valid_min'][0]
        value = vari[()] * 1.
        value[(value < valid_min) | (value > valid_max)] = np.nan
        vaa = value * scale + offset
        f.close()

        return [Rrs, aer_model_ratio, La_nirl, F0_nirl, lat, lon, sza, vza, saa, vaa, aer_model_max,
                aer_model_min]


def sensor(sensor: str):
    def s3a_olci():
        nirl = 865
        sensor_info = read_sensorinfo(sensorinfo_file="../share/olci/s3a/msl12_sensor_info.dat")
        [bands, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
         oobwv, ooblw, wed, waph] = sensor_info
        return bands, F0, nirl

    def s3b_olci():
        nirl = 865
        sensor_info = read_sensorinfo(sensorinfo_file="../share/olci/s3b/msl12_sensor_info.dat")
        [bands, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
         oobwv, ooblw, wed, waph] = sensor_info
        return bands, F0, nirl

    def terra_modis():
        nirl = 869
        sensor_info = read_sensorinfo(sensorinfo_file="../share/modis/terra/msl12_sensor_info.dat")
        [bands, F0, Tau_r, k_oz, t_co2, k_no2, a_h2o, b_h2o, c_h2o, d_h2o, e_h2o, f_h2o, g_h2o, awhite, aw, bbw,
         oobwv, ooblw, wed, waph] = sensor_info
        return bands, F0, nirl

    if sensor == "s3a_olci":
        return s3a_olci()
    elif sensor == "s3b_olci":
        return s3b_olci()
    elif sensor == "terra_modis":
        return terra_modis()
