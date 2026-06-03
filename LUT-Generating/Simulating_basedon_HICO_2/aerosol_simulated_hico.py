# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:20:32 2022
根据hico的气溶胶查找表计算其他卫星的气溶胶查找表
@author: Jilin men，, wenkai li修改
"""
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
import datetime
import os

def read_hico_aerosol(path):
    '''
    读取hico的气溶胶LUTs
    '''
    dt = SD(path, SDC.READ)
    wave = (dt.select('wave')).get()  # 1d
    scatt = (dt.select('scatt')).get()  # fixed

    albedo = (dt.select('albedo')).get()  # 1d
    extc = (dt.select('extc')).get()  # 1d
    angstrom = (dt.select('angstrom')).get()  # 1d
    phase = (dt.select('phase')).get()  # 2d

    solz = (dt.select('solz')).get()  # fixed
    senz = (dt.select('senz')).get()  # fixed
    phi = (dt.select('phi')).get()  # fixed

    acost = (dt.select('acost')).get()  # 4d, 1st 二项系数
    bcost = (dt.select('bcost')).get()
    ccost = (dt.select('ccost')).get()

    dtran_wave = (dt.select('dtran_wave')).get()
    dtran_theta = (dt.select('dtran_theta')).get()

    dtran_a = (dt.select('dtran_a')).get()
    dtran_b = (dt.select('dtran_b')).get()
    dtran_a0 = (dt.select('dtran_a0')).get()
    dtran_b0 = (dt.select('dtran_b0')).get()
    att_sd = getattr(dt, 'Size Distribution')

    return wave, scatt, albedo, extc, angstrom, phase, solz, senz, \
           phi, acost, bcost, ccost, dtran_wave, dtran_theta, dtran_a, dtran_b, dtran_a0, dtran_b0, att_sd


def read_rsr(file, sensorid):
    '''
    读取传感器的光谱响应函数
    '''
    rsr = pd.read_excel(io=file, sheet_name=sensorid, header=0, index_col=0)
    wave_rsr = rsr.index
    wave_target = rsr.columns
    return rsr, wave_rsr, wave_target


def new_aerosol_1d(wave_target, rsr, wave_hico, albedo, extc, angstrom):
    '''
    s1:将hico的波长四舍五入
    s2:删除不在hico光谱范围内的波段,[352,1080]
    s3:提取各个波段的响应波长，响应度大于0.01
    s4:根据响应波长的响应程度和范围计算hico波段的平均值
    '''
    # wave_target = np.array(wave_target.tolist(),'int')
    length = wave_target.__len__()
    albedo_new = np.empty(length)
    extc_new = np.empty(length)
    angstrom_new = np.empty(length)
    wave_hico = np.array(wave_hico, 'int')
    for i in range(len(wave_target)):
        # if int(wave_target[i])<1080:
        response_range = rsr[wave_target[i]][rsr[wave_target[i]] > 0.001]
        response_wave = response_range.index.values
        response_val = response_range.values

        # albedo
        albedo_a = albedo[np.isin(wave_hico, response_wave)]
        albedo_b = response_val[np.isin(response_wave, wave_hico)]
        albedo_new[i] = np.sum(albedo_a * albedo_b) / np.sum(albedo_b)

        # extc
        extc_a = extc[np.isin(wave_hico, response_wave)]
        extc_b = response_val[np.isin(response_wave, wave_hico)]
        extc_new[i] = np.sum(extc_a * extc_b) / np.sum(extc_b)

        # angstrom
        angstrom_a = angstrom[np.isin(wave_hico, response_wave)]
        angstrom_b = response_val[np.isin(response_wave, wave_hico)]
        angstrom_new[i] = np.sum(angstrom_a * angstrom_b) / np.sum(angstrom_b)

        # else:
        #     print('Wavelength is larger than 1080 nm')

    return np.array(albedo_new, 'float32'), np.array(extc_new, 'float32'), np.array(angstrom_new, 'float32')


def new_aerosol_2d(wave_target, rsr, wave_hico, phase, dtran_a, dtran_b, dtran_a0, dtran_b0):
    '''
    计算两维参量
    '''
    # wave_target = np.array(wave_target.tolist(),'int')
    length = np.sum(np.array(wave_target.tolist(), 'int') < 1080)
    phase_new = np.empty((length, 75))
    dtran_a_new = np.empty((length, 33))
    dtran_b_new = np.empty((length, 33))
    dtran_a0_new = np.empty((length, 33))
    dtran_b0_new = np.empty((length, 33))
    wave_hico = np.array(wave_hico, 'int')
    for i in range(len(wave_target)):
        if int(wave_target[i]) < 1080:
            response_wave = rsr[wave_target[i]][rsr[wave_target[i]] > 0.1].index.tolist()
            response_wave = np.round(response_wave)
            response_val = np.array(rsr[wave_target[i]][rsr[wave_target[i]] > 0.1])

            # phase
            phase_a = phase[np.isin(wave_hico, response_wave), :]
            phase_b = response_val[np.isin(response_wave, wave_hico)][:, np.newaxis]
            phase_b = phase_b / np.sum(phase_b)
            phase_new[i, :] = np.sum(phase_a * phase_b, axis=0)

            # dtran_a
            dtran_a_a = dtran_a[np.isin(wave_hico, response_wave), :]
            dtran_a_b = response_val[np.isin(response_wave, wave_hico)][:, np.newaxis]
            dtran_a_b = dtran_a_b / np.sum(dtran_a_b)
            dtran_a_new[i, :] = np.sum(dtran_a_a * dtran_a_b, axis=0)

            # dtran_b
            dtran_b_a = dtran_b[np.isin(wave_hico, response_wave), :]
            dtran_b_b = response_val[np.isin(response_wave, wave_hico)][:, np.newaxis]
            dtran_b_b = dtran_b_b / np.sum(dtran_b_b)
            dtran_b_new[i, :] = np.sum(dtran_b_a * dtran_b_b, axis=0)

            # dtran_a0
            dtran_a0_a = dtran_a0[np.isin(wave_hico, response_wave), :]
            dtran_a0_b = response_val[np.isin(response_wave, wave_hico)][:, np.newaxis]
            dtran_a0_b = dtran_a0_b / np.sum(dtran_a0_b)
            dtran_a0_new[i, :] = np.sum(dtran_a0_a * dtran_a0_b, axis=0)

            # dtran_b0
            dtran_b0_a = dtran_b0[np.isin(wave_hico, response_wave), :]
            dtran_b0_b = response_val[np.isin(response_wave, wave_hico)][:, np.newaxis]
            dtran_b0_b = dtran_b0_b / np.sum(dtran_b0_b)
            dtran_b0_new[i, :] = np.sum(dtran_b0_a * dtran_b0_b, axis=0)

        else:
            print('Wavelength is larger than 1080 nm')

    return np.array(phase_new, 'float32'), np.array(dtran_a_new, 'float32'), np.array(dtran_b_new, 'float32'), \
           np.array(dtran_a0_new, 'float32'), np.array(dtran_b0_new, 'float32')


def new_aerosol_4d(wave_target, rsr, wave_hico, acost, bcost, ccost):
    '''
    计算四维参量,acost,bcost,ccost
    '''
    length = np.sum(np.array(wave_target.tolist(), 'int') < 1080)
    acost_new = np.empty((length, 33, 19, 35))
    bcost_new = np.empty((length, 33, 19, 35))
    ccost_new = np.empty((length, 33, 19, 35))
    wave_hico = np.array(wave_hico, 'int')
    for i in range(len(wave_target)):
        if int(wave_target[i]) < 1080:
            response_wave = rsr[wave_target[i]][rsr[wave_target[i]] > 0.1].index.tolist()
            response_wave = np.round(response_wave)
            response_val = np.array(rsr[wave_target[i]][rsr[wave_target[i]] > 0.1])

            # acost
            acost_a = acost[np.isin(wave_hico, response_wave), :, :, :]
            acost_b = response_val[np.isin(response_wave, wave_hico)]
            acost_b = acost_b / np.sum(acost_b)
            a = [acost_b[i] * acost_a[i, :, :, :] for i in range(len(acost_b))]
            for j in range(len(a)):
                if j == 0:
                    acost_new[i, :, :, :] = a[j]
                else:
                    acost_new[i, :, :, :] = acost_new[i, :, :, :] + a[j]

            # bcost
            bcost_a = bcost[np.isin(wave_hico, response_wave), :, :, :]
            bcost_b = response_val[np.isin(response_wave, wave_hico)]
            bcost_b = bcost_b / np.sum(bcost_b)
            a = [bcost_b[i] * bcost_a[i, :, :, :] for i in range(len(bcost_b))]
            for j in range(len(a)):
                if j == 0:
                    bcost_new[i, :, :, :] = a[j]
                else:
                    bcost_new[i, :, :, :] = bcost_new[i, :, :, :] + a[j]

            # ccost
            ccost_a = ccost[np.isin(wave_hico, response_wave), :, :, :]
            ccost_b = response_val[np.isin(response_wave, wave_hico)]
            ccost_b = ccost_b / np.sum(ccost_b)
            a = [ccost_b[i] * ccost_a[i, :, :, :] for i in range(len(ccost_b))]
            for j in range(len(a)):
                if j == 0:
                    ccost_new[i, :, :, :] = a[j]
                else:
                    ccost_new[i, :, :, :] = ccost_new[i, :, :, :] + a[j]

        else:
            print('Wavelength is larger than 1080 nm')

    return np.array(acost_new, 'float32'), np.array(bcost_new, 'float32'), np.array(ccost_new, 'float32')


def run_main(rsr_path,path_hico, path_out, sensorid):
    models = ['r30f95', 'r30f80', 'r30f50', 'r30f30', 'r30f20', 'r30f10', 'r30f05', 'r30f02',
              'r30f01', 'r30f00', 'r50f95', 'r50f80', 'r50f50', 'r50f30', 'r50f20', 'r50f10',
              'r50f05', 'r50f02', 'r50f01', 'r50f00', 'r70f95', 'r70f80', 'r70f50', 'r70f30',
              'r70f20', 'r70f10', 'r70f05', 'r70f02', 'r70f01', 'r70f00', 'r75f95', 'r75f80',
              'r75f50', 'r75f30', 'r75f20', 'r75f10', 'r75f05', 'r75f02', 'r75f01', 'r75f00',
              'r80f95', 'r80f80', 'r80f50', 'r80f30', 'r80f20', 'r80f10', 'r80f05', 'r80f02',
              'r80f01', 'r80f00', 'r85f95', 'r85f80', 'r85f50', 'r85f30', 'r85f20', 'r85f10',
              'r85f05', 'r85f02', 'r85f01', 'r85f00', 'r90f95', 'r90f80', 'r90f50', 'r90f30',
              'r90f20', 'r90f10', 'r90f05', 'r90f02', 'r90f01', 'r90f00', 'r95f95', 'r95f80',
              'r95f50', 'r95f30', 'r95f20', 'r95f10', 'r95f05', 'r95f02', 'r95f01', 'r95f00']
    '''修改RSR路径'''
    # rsr_path = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\RSR\RSR.xlsx'  # 光谱响应函数
    # satellite = 'sdgsat1mii'  # 模拟的传感器名称
    for i in range(len(models)):
        '''修改路径'''
        # lut_path = r'D:\researchProject_lwk\atmospheric_correction\seadas\share\hico\aerosol\aerosol_hico_' + \
        #            models[i] + 'v01.hdf'  # hico查找表路径
        # out_path = r'D:\researchProject_lwk\atmospheric_correction\oceancolor_acnirv2\share\sdgsat1mii\aerosol\aerosol_' + satellite + '_' + \
        #            models[i] + '.hdf'  # 模拟查找表的输出路径
        lut_path=path_hico+os.sep+'aerosol_hico_' + models[i] + 'v01.hdf'  # hico查找表路径
        out_path = path_out+os.sep+'aerosol_' + sensorid + '_' + models[i] + 'v01.hdf'  # 模拟查找表的输出路径

        wave, scatt, albedo, extc, angstrom, phase, solz, senz, \
        phi, acost, bcost, ccost, dtran_wave, dtran_theta, dtran_a, dtran_b, dtran_a0, dtran_b0, att_sd = \
            read_hico_aerosol(lut_path)
        rsr, wave_rsr, wave_target = read_rsr(rsr_path,sensorid)

        albedo_new, extc_new, angstrom_new = new_aerosol_1d(wave_target, rsr, wave, albedo, extc, angstrom)
        phase_new, dtran_a_new, dtran_b_new, dtran_a0_new, dtran_b0_new = new_aerosol_2d(wave_target, rsr, wave, phase,
                                                                                         dtran_a, dtran_b, dtran_a0,
                                                                                         dtran_b0)
        acost_new, bcost_new, ccost_new = new_aerosol_4d(wave_target, rsr, wave, acost, bcost, ccost)

        '''save'''
        hdffile = SD(out_path, SDC.WRITE | SDC.CREATE)
        hdffile.attr('Title').set(SDC.CHAR, 'Aerosol Model Data for ' + sensorid)
        hdffile.attr('Model Name').set(SDC.CHAR, models[i])
        hdffile.attr('Version').set(SDC.CHAR, '01')
        hdffile.attr('Relative Humidity').set(SDC.FLOAT32, float(models[i][1:3]))
        hdffile.attr('Size Distribution').set(SDC.INT16, att_sd)
        hdffile.attr('Number of Wavelengths').set(SDC.INT16, len(wave_target))
        hdffile.attr('Number of Scattering Angles').set(SDC.INT16, len(scatt))
        hdffile.attr('Number of Solar Zenith Angles').set(SDC.INT16, len(solz))
        hdffile.attr('Number of View Zenith Angles').set(SDC.INT16, len(senz))
        hdffile.attr('Number of Relative Azimuth Angles').set(SDC.INT16, len(phi))
        hdffile.attr('Number of Diffuse Transmittance Wavelengths').set(SDC.INT16, len(wave_target))
        hdffile.attr('Number of Diffuse Transmittance Zenith Angles').set(SDC.INT16, len(dtran_theta))
        hdffile.attr('Creation Date').set(SDC.CHAR, str(datetime.datetime.now()))
        hdffile.attr('Created by').set(SDC.CHAR, 'Jilin Men and Wenkai Li based on  HICO')

        # create dataset
        wave_target = wave_target.astype('float32')
        wave_target = wave_target[wave_target <= 1080].to_numpy().astype('float32')

        d1 = hdffile.create('wave', SDC.FLOAT32, wave_target.shape)
        d1.long_name = 'wavelengths'
        d1.units = 'nm'
        dim1 = d1.dim(0)
        dim1.setname('nwave')
        d1[:] = wave_target

        d2 = hdffile.create('scatt', SDC.FLOAT32, scatt.shape)
        d2.long_name = 'scattering angles'
        d2.units = 'degrees'
        d2_dim1 = d2.dim(0)
        d2_dim1.setname('nscatt')
        d2[:] = scatt

        d3 = hdffile.create('albedo', SDC.FLOAT32, albedo_new.shape)
        d3.long_name = 'single scattering albedo'
        d3.units = 'dimensionless'
        d3_dim1 = d3.dim(0)
        d3_dim1.setname('nwave')
        d3[:] = albedo_new

        d4 = hdffile.create('extc', SDC.FLOAT32, extc_new.shape)
        d4.long_name = 'extinction coefficient'
        d4.units = 'dimensionless'
        d4_dim1 = d4.dim(0)
        d4_dim1.setname('nwave')
        d4[:] = extc_new

        d5 = hdffile.create('angstrom', SDC.FLOAT32, angstrom_new.shape)
        d5.long_name = 'angstrom coefficient'
        d5.units = 'dimensionless'
        d5_dim1 = d5.dim(0)
        d5_dim1.setname('nwave')
        d5[:] = angstrom_new

        d6 = hdffile.create('phase', SDC.FLOAT32, phase_new.shape)
        d6.long_name = 'volume scattering function'
        d6.units = 'dimensionless'
        d6_dim1 = d6.dim(0)
        d6_dim2 = d6.dim(1)
        d6_dim1.setname('nwave')
        d6_dim2.setname('nscatt')
        d6[:] = phase_new

        d7 = hdffile.create('solz', SDC.FLOAT32, solz.shape)
        d7.long_name = 'solar zenith angles'
        d7.units = 'degrees'
        d7_dim1 = d7.dim(0)
        d7_dim1.setname('nsolz')
        d7[:] = solz

        d8 = hdffile.create('senz', SDC.FLOAT32, senz.shape)
        d8.long_name = 'sensor view zenith angles'
        d8.units = 'degrees'
        d8_dim1 = d8.dim(0)
        d8_dim1.setname('nsenz')
        d8[:] = senz

        d9 = hdffile.create('phi', SDC.FLOAT32, phi.shape)
        d9.long_name = 'relative azimuth angles'
        d9.units = 'degrees'
        d9_dim1 = d9.dim(0)
        d9_dim1.setname('nphi')
        d9[:] = phi

        d10 = hdffile.create('acost', SDC.FLOAT32, acost_new.shape)
        d10.long_name = '1st quadratic coefficient of SS to MS function'
        d10.units = 'dimensionless'
        d10_dim1 = d10.dim(0)
        d10_dim2 = d10.dim(1)
        d10_dim3 = d10.dim(2)
        d10_dim4 = d10.dim(3)
        d10_dim1.setname('nwave')
        d10_dim2.setname('nsolz')
        d10_dim3.setname('nphi')
        d10_dim4.setname('nsenz')
        d10[:] = acost_new

        d11 = hdffile.create('bcost', SDC.FLOAT32, bcost_new.shape)
        d11.long_name = '2st quadratic coefficient of SS to MS function'
        d11.units = 'dimensionless'
        d11_dim1 = d11.dim(0)
        d11_dim2 = d11.dim(1)
        d11_dim3 = d11.dim(2)
        d11_dim4 = d11.dim(3)
        d11_dim1.setname('nwave')
        d11_dim2.setname('nsolz')
        d11_dim3.setname('nphi')
        d11_dim4.setname('nsenz')
        d11[:] = bcost_new

        d12 = hdffile.create('ccost', SDC.FLOAT32, ccost_new.shape)
        d12.long_name = '3st quadratic coefficient of SS to MS function'
        d12.units = 'dimensionless'
        d12_dim1 = d12.dim(0)
        d12_dim2 = d12.dim(1)
        d12_dim3 = d12.dim(2)
        d12_dim4 = d12.dim(3)
        d12_dim1.setname('nwave')
        d12_dim2.setname('nsolz')
        d12_dim3.setname('nphi')
        d12_dim4.setname('nsenz')
        d12[:] = ccost_new

        d13 = hdffile.create('dtran_wave', SDC.FLOAT32, wave_target.shape)
        d13.long_name = 'wavelengths of the diffuse transmittance coeffs'
        d13.units = 'nm'
        d13_dim1 = d13.dim(0)
        d13_dim1.setname('dtran_nwave')
        d13[:] = wave_target

        d14 = hdffile.create('dtran_theta', SDC.FLOAT32, dtran_theta.shape)
        d14.long_name = 'zenith angles of the diffuse transmittance coeffs'
        d14.units = 'degrees'
        d14_dim1 = d14.dim(0)
        d14_dim1.setname('dtran_ntheta')
        d14[:] = dtran_theta

        d15 = hdffile.create('dtran_a', SDC.FLOAT32, dtran_a_new.shape)
        d15.long_name = 'a coefficient of diffuse sensor transmittance'
        d15.units = 'dimensionless'
        d15_dim1 = d15.dim(0)
        d15_dim2 = d15.dim(1)
        d15_dim1.setname('dtran_nwave')
        d15_dim2.setname('dtran_ntheta')
        d15[:] = dtran_a_new

        d16 = hdffile.create('dtran_b', SDC.FLOAT32, dtran_b_new.shape)
        d16.long_name = 'b coefficient of diffuse sensor transmittance'
        d16.units = 'dimensionless'
        d16_dim1 = d16.dim(0)
        d16_dim2 = d16.dim(1)
        d16_dim1.setname('dtran_nwave')
        d16_dim2.setname('dtran_ntheta')
        d16[:] = dtran_b_new

        d17 = hdffile.create('dtran_a0', SDC.FLOAT32, dtran_a0_new.shape)
        d17.long_name = 'a coefficient of diffuse solar transmittance'
        d17.units = 'dimensionless'
        d17_dim1 = d17.dim(0)
        d17_dim2 = d17.dim(1)
        d17_dim1.setname('dtran_nwave')
        d17_dim2.setname('dtran_ntheta')
        d17[:] = dtran_a0_new

        d18 = hdffile.create('dtran_b0', SDC.FLOAT32, dtran_b0_new.shape)
        d18.long_name = 'b coefficient of diffuse solar transmittance'
        d18.units = 'dimensionless'
        d18_dim1 = d18.dim(0)
        d18_dim2 = d18.dim(1)
        d18_dim1.setname('dtran_nwave')
        d18_dim2.setname('dtran_ntheta')
        d18[:] = dtran_b0_new

        d1.endaccess()
        d2.endaccess()
        d3.endaccess()
        d4.endaccess()
        d5.endaccess()
        d6.endaccess()
        d7.endaccess()
        d8.endaccess()
        d9.endaccess()
        d10.endaccess()
        d11.endaccess()
        d12.endaccess()
        d13.endaccess()
        d14.endaccess()
        d15.endaccess()
        d16.endaccess()
        d17.endaccess()
        d18.endaccess()
        hdffile.end()


if __name__ == '__main__':
    run_main()

