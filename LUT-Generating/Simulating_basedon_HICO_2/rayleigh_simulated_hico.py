# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:20:32 2022
根据hico的瑞丽查找表计算其他卫星的瑞丽查找表
@author: Jilin men, wenkai li修改
"""
import numpy as np
import h5py
import pandas as pd
from pyhdf.SD import SD, SDC
import datetime
import glob
import os


def read_hico_rayleigh(path):
    dt = SD(path, SDC.READ)
    taur = (dt.select('taur')).get()  # 1d
    depol = (dt.select('depol')).get()
    senz = (dt.select('senz')).get()
    solz = (dt.select('solz')).get()
    sigma = (dt.select('sigma')).get()
    i_ray = (dt.select('i_ray')).get()
    q_ray = (dt.select('q_ray')).get()
    u_ray = (dt.select('u_ray')).get()

    return taur, depol, senz, solz, sigma, i_ray, q_ray, u_ray


def read_rsr(file,sensorid):
    '''
    读取传感器的光谱响应函数
    '''
    # rsr = pd.read_excel(io=file, sheet_name=sensorid, header=0, index_col=0)
    # wave_rsr = rsr.index
    # wave_target = rsr.columns

    rsr = pd.read_excel(io=file, sheet_name=sensorid, header=0, index_col=0)
    # wave_rsr = rsr.index
    # wave_target = rsr.columns
    wave_target = rsr.columns[1::2]
    return rsr, wave_rsr, wave_target


def run_main(path_rsr, path_hico, out_path, sensorid):
    '''
        s1 读取光谱响应函数
        s2 根据光谱响应函数提取响应波段范围和响应值
        s3 响应波段与hico波段进行匹配
        s4 读取hico波段瑞丽文件，积分计算
        '''
    '''修改路径'''
    # path_rsr = r'D:\researchProject_lwk\git_repository\atmosphericCorrection\oceancolorACnirV21\RSR\RSR.xlsx'  # 光谱响应函数路径
    # path_hico = r'D:\researchProject_lwk\git_repository\atmosphericCorrection\seadas\share\hico\rayleigh'  # HICO rayleigh查找表路径
    # out_path = r'D:\researchProject_lwk\git_repository\atmosphericCorrection\oceancolorACnirV21\share\sdgsatmii\rayleigh'  # 模拟查找表的输出路径
    # satellite = 'sdgsat1mii'  # 模拟的传感器名称
    rsr, wave_rsr, wave_target = read_rsr(path_rsr,sensorid)

    hico_bands = []
    files = glob.glob(os.path.join(path_hico, '*.hdf'))
    hico_bands = np.array([int(os.path.basename(i).split("_", -1)[2]) for i in files])

    for i in range(wave_target.__len__()):  # 小于1080nm的波段
        # 响应范围
        wave_rsr = rsr.index
        wave_rsr = rsr.iloc[:, 2 * i]
        srf_band = spectrum_response_function.iloc[:, 2 * i + 1]
        response_range =

        response_range = rsr[wave_target[i]][rsr[wave_target[i]] > 0.1]
        response_wave = response_range.index.values
        response_val = response_range.values

        hico_response = hico_bands[np.isin(hico_bands, response_wave)]
        response_val = response_val[np.isin(response_wave, hico_bands)]
        # 加载hico LUTs
        for j in range(len(hico_response)):
            path_hicoband = os.path.join(path_hico, 'rayleigh_hico_' + str(hico_response[j]) + '_iqu.hdf')
            dt = SD(path_hicoband, SDC.READ)
            if j == 0:
                taur = (dt.select('taur')).get() * response_val[j] / np.sum(response_val)
                depol = (dt.select('depol')).get() * response_val[j] / np.sum(response_val)
                i_ray = (dt.select('i_ray')).get() * response_val[j] / np.sum(response_val)
                q_ray = (dt.select('q_ray')).get() * response_val[j] / np.sum(response_val)
                u_ray = (dt.select('u_ray')).get() * response_val[j] / np.sum(response_val)
            else:
                taur = taur + ((dt.select('taur')).get()) * response_val[j] / np.sum(response_val)
                depol = depol + ((dt.select('depol')).get()) * response_val[j] / np.sum(response_val)
                i_ray = i_ray + (dt.select('i_ray')).get() * response_val[j] / np.sum(response_val)
                q_ray = q_ray + (dt.select('q_ray')).get() * response_val[j] / np.sum(response_val)
                u_ray = u_ray + (dt.select('u_ray')).get() * response_val[j] / np.sum(response_val)

        senz = (dt.select('senz')).get()
        solz = (dt.select('solz')).get()
        sigma = (dt.select('sigma')).get()

        # save
        hdffile = SD(os.path.join(out_path, 'rayleigh_' + sensorid + '_' + str(wave_target[i]) + '_iqu.hdf'),
                     SDC.WRITE | SDC.CREATE)
        hdffile.Title = 'Rayleigh Radiance Tables for ' + sensorid + ' at ' + str(wave_target[i]) + ' nm'
        hdffile.Creation_Date = str(datetime.datetime.now())
        hdffile.Created_by = 'Jinlin Men, Wenkai li'

        # create dataset
        # wave_target = wave_target.astype('float32')
        # wave_target = wave_target[wave_target<1080]

        d1 = hdffile.create('taur', SDC.FLOAT32, taur.shape)
        d1.long_name = 'Optical Thickness'
        d1.slope = 1.0
        d1.intercept = 0.0
        d1.units = 'dimensionless'
        dim1 = d1.dim(0)
        dim1.setname('nlambda')
        d1[:] = taur

        d2 = hdffile.create('depol', SDC.FLOAT32, depol.shape)
        d2.long_name = 'Depolarization Factor'
        d2.slope = 1.0
        d2.intercept = 0.0
        d2.units = 'dimensionless'
        dim1 = d2.dim(0)
        dim1.setname('nlambda')
        d2[:] = depol

        d3 = hdffile.create('senz', SDC.FLOAT32, senz.shape)
        d3.long_name = 'Sensor Zenith Angles'
        d3.slope = 1.0
        d3.intercept = 0.0
        d3.units = 'degrees'
        # dim1 = d3.dim(0)
        # dim1.setname('nrad_ray')
        d3[:] = senz

        d4 = hdffile.create('solz', SDC.FLOAT32, solz.shape)
        d4.long_name = 'Solar Zenith Angles'
        d4.slope = 1.0
        d4.intercept = 0.0
        d4.units = 'degrees'
        # dim1 = d4.dim(0)
        # dim1.setname('nrad_ray')
        d4[:] = solz

        d5 = hdffile.create('sigma', SDC.FLOAT32, sigma.shape)
        d5.long_name = 'sigma of Wind Speed'
        d5.slope = 1.0
        d5.intercept = 0.0
        d5.units = 'sqrt(m/s)'
        # dim1 = d5.dim(0)
        # dim1.setname('nwind_ray')
        # d5[:] = (sigma/0.0731)**2
        d5[:] = sigma

        d6 = hdffile.create('i_ray', SDC.FLOAT32, i_ray.shape)
        d6.long_name = 'Rayleigh Reflectance Coefficients for I-compoent'
        d6.slope = 1.0
        d6.intercept = 0.0
        d6.units = 'dimensionless'
        d6[:] = i_ray

        d7 = hdffile.create('q_ray', SDC.FLOAT32, q_ray.shape)
        d7.long_name = 'Rayleigh Reflectance Coefficients for Q-compoent'
        d7.slope = 1.0
        d7.intercept = 0.0
        d7.units = 'dimensionless'
        d7[:] = q_ray

        d8 = hdffile.create('u_ray', SDC.FLOAT32, u_ray.shape)
        d8.long_name = 'Rayleigh Reflectance Coefficients for U-compoent'
        d8.slope = 1.0
        d8.intercept = 0.0
        d8.units = 'dimensionless'
        d8[:] = u_ray

        d1.endaccess()
        d2.endaccess()
        d3.endaccess()
        d4.endaccess()
        d5.endaccess()
        d6.endaccess()
        d7.endaccess()
        d8.endaccess()
        hdffile.end()


if __name__ == '__main__':
    hico_dir = r"D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\share\hico"
    base_dir = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2'
    rsr_dir = base_dir + os.sep + "RSR"
    rsr_infile = rsr_dir + os.sep + 'RSR.xlsx'
    infofile_taur = rsr_dir + os.sep + "taur.txt"
    infofile_ozone = rsr_dir + os.sep + 'Ozoneattenuationcoefficients'
    infofile_no2 = rsr_dir + os.sep + 'NO2absorption'
    sensorid = "sdgsat1mii"  # "goci"
    lut_target_dir = base_dir + os.sep + "share" + os.sep + sensorid
    out_path_ray = lut_target_dir + os.sep + "rayleigh"
    if not os.path.exists(out_path_ray):
        os.mkdir(out_path_ray)
    run_main(path_rsr=rsr_infile,
             path_hico=hico_dir + os.sep + "rayleigh",
             out_path=out_path_ray,
             sensorid=sensorid)
