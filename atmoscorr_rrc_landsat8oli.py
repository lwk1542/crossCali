# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: atmoscorr_rrc_landsat8oli.py
@time: 2022/6/1 17:34
@desc:
"""

import datetime
import os

import h5py
import numpy as np

from atmospheric_correction.oceancolor_acnirv2.sharepy import readfile
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import atmosphericParameter, gas_transmittance
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import whitecap_rad, general, rayleigh_rad

if __name__ == '__main__':

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    south, north, west, east = None, None, None, None  # 20, 41, 116, 130  # 21, 22, 125, 126 # 15, 24.22, 110, 121.5
    sensorID = 'lc08oli'
    sensor_alt = 780
    nonzeroNIR = "want_nirLw"
    filepath = r"F:\cali_spatial_vari\OLI"  # 数据路径

    files = general.get_filelistv2('LC08_L1TP', "_T1.tar.gz", path=filepath, mode='all')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for num, infile in enumerate(files):

        starttime = datetime.datetime.now()
        print('Processing:' + str(num + 1) + '/' + str(files.__len__()) + ' image: ' + os.path.basename(infile))
        outfile = infile.replace(".tar.gz", '_L2A_iterNIR.h5')
        if os.path.exists(outfile):
            if os.path.getsize(outfile) / 1024. / 1024 > 10:
                continue
            else:
                os.remove(outfile)
        else:
            pass

        [lut_path, sensor_info, image_info] = readfile.get_info(infile=infile, sensorID=sensorID, mode="ac",
                                                                north=north, south=south,
                                                                west=west, east=east)
        (rayleigh_lut_path, aerosol_lut_path) = lut_path
        (bands, Fo, Tau_r, k_oz, t_co2, k_no2, Zia_table, awhite, aw, bbw, oobwv, ooblw, wed, waph) = sensor_info
        (sza, vza, saa, vaa, lat, lon, Lt, year, month, day, num_443, num_490, num_520, num_555, num_670,
         nirs_num, nirl_num, nwvis, red) = image_info
        fqfile = r"share" + os.sep + "common" + os.sep + 'morel_fq.h5'

        # 2. 下载气象数据，根据海洋卫星经纬度插值出相应的气象参数：风速、气压、 臭氧，其它如水汽柱也可输出
        date = datetime.datetime.strptime(str(year) + str(month) + str(day), '%Y%m%d')
        doy = int(date.strftime('%j'))
        A, B, C, D, E = 1.00014, 0.01671, 0.9856002831, 3.452868, 360.
        fsol = 1. / ((A - B * np.cos(2. * np.pi * (C * doy - D) / E) - 0.000014 * np.cos(
            4. * np.pi * (C * doy - D) / E)) ** 2)
        print('correcting coefficient of solar-earth distance: ' + str(fsol)[0:5])
        FoBAR = Fo * fsol
        Fo_ = np.zeros(shape=(1, 1, Fo.size))
        Fo_[0, 0, :] = FoBAR

        print('load atmospheric parameters: pressure, O3, NO2, water vapor, reality humidity, wind etc:...')
        winds_peed, winddirection, pressure, o3, rh, water_vapor, strat_no2, trop_no2, taua = \
            atmosphericParameter.get(Lon=lon, Lat=lat, year=year, month=month, day=day, time='03:00')

        # 3. 瑞利光学厚度的气压校正
        # /* Pressure correct the Rayleigh optical thickness */
        factor = pressure / 1013.25
        taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * Tau_r.reshape(1, 1, -1)

        # 4. 臭氧透过率校正，臭氧从欧洲下载，详情见函数内部。Mobely给的消光截面单位是错误的
        print('computing gas absorbing transmittance: o3, no2, co2, h2o...')
        # 对于区域范围小，空间分辨率高的影像，气体吸收计算单个值就行了
        sza_temp = np.array([[np.nanmean(sza)]])
        vza_temp = np.array([[np.nanmean(vza)]])
        o3_temp = np.array([[np.nanmean(o3)]])
        strat_no2_temp = np.array([[np.nanmean(strat_no2)]])
        trop_no2_temp = np.array([[np.nanmean(trop_no2)]])
        water_vapor_temp = np.array([[np.nanmean(water_vapor)]])
        tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=sza_temp, vza=vza_temp, koz=k_oz,
                                                                       concentration=o3_temp)
        tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=k_no2, sza=sza_temp, vza=vza_temp,
                                                                          strat_no2=strat_no2_temp,
                                                                          trop_no2=trop_no2_temp)
        tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=t_co2, sza=sza_temp, vza=vza_temp)
        tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=water_vapor_temp, sza=sza_temp,
                                                                          vza=vza_temp, zia_table=Zia_table)

        tg_sol = tg_solar_o3 * tg_solar_no2 * tg_solar_co2 * tg_solar_h2o  # 其它吸收暂时不考虑
        tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_sensor_co2 * tg_sensor_h2o

        # /* Correct for ozone absorption.  We correct for inbound and outbound here, then we put the inbound back
        # when computing Lw.*/ Ltemp[ib] = Ltemp[ib] / l1rec->tg_sol[ipb] / l1rec->tg_sen[ipb];
        Ltemp = Lt / tg_sen / tg_sol  # 上下和下行透过率

        # 卷云和极化校正：没做
        # / *Apply polarization correction * /

        # 5. 移除白帽反射
        print('computing whitecap radiance...')
        rho_wc = whitecap_rad.calculate(U10=winds_peed, bands=bands)
        t_sen = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))
        t_sol = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))
        tLf = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))

        for i in range(bands.size):
            t_sen[:, :, i] = np.exp(-0.5 * taur[:, :, i] / np.cos(vza * np.pi / 180))
            t_sol[:, :, i] = np.exp(-0.5 * taur[:, :, i] / np.cos(sza * np.pi / 180))
            tLf[:, :, i] = rho_wc[:, :, i] * t_sol[:, :, i] * t_sen[:, :, i] * FoBAR[i] * np.cos(
                sza * np.pi / 180) / np.pi
        Ltemp = Ltemp - tLf

        # 6. 移除瑞利贡献
        print('computing rayleigh scattering radaince...')
        lr = rayleigh_rad.rayleigh(rayleigh_lut_path=rayleigh_lut_path, sza=sza, vza=vza, saa=saa, vaa=vaa,
                                   F0=FoBAR, windspeed=winds_peed, pressure=pressure)
        airmass1 = 1 / np.cos(sza * np.pi / 180) + 1 / np.cos(vza * np.pi / 180)
        # 7.氧气校正：只需要对750nm做，直接沿用了seawifs的校正系数,仅需要对特定传感器校正
        a_o2 = gas_transmittance.oxygen_ray(airmass1)
        t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass1)
        lr[:, :, nirs_num] = lr[:, :, nirs_num] * a_o2  #
        scaleRayleigh = 1.0 - np.exp(-sensor_alt / 10)
        lr = lr * scaleRayleigh
        Lrc = (Ltemp - lr)
        Rrc = np.pi * Lrc / Fo_ / np.cos(np.pi * sza.reshape(sza.shape[0], sza.shape[1], 1) / 180)
        Ltemp = Ltemp - lr


        # 写入文件
        f_new = h5py.File(outfile, 'a')
        f_new.attrs.create('input file', os.path.basename(infile), shape=(1,), dtype='S80')
        f_new.attrs.create('product time', starttime.strftime("%m/%d/%Y, %H:%M:%S"), shape=(1,), dtype='S80')
        f_new.attrs.create('method', 'iter NIR atmospheric correction', shape=(1,), dtype='S26')
        (rows, columns) = Ltemp[:, :, 0].shape
        GeoData = f_new.create_group("geophysical_data")
        for k, band in enumerate(bands):
            # GeoData.create_dataset('Lt_' + str(band), (rows, columns), dtype='f', data=Lt[:, :, k])
            # GeoData.create_dataset('Lr_' + str(band), (rows, columns), dtype='f', data=lr[:, :, k])
            # GeoData.create_dataset('tLf_' + str(band), (rows, columns), dtype='f', data=tLf[:, :, k])
            # GeoData.create_dataset('TLg_' + str(band), (rows, columns), dtype='f', data=TLg[:, :, k])
            # GeoData.create_dataset('La_' + str(band), (rows, columns), dtype='f', data=La[:, :, k])
            mask = np.nan_to_num(Ltemp[:, :, k], nan=-999, posinf=-999, neginf=-999)
            mask[mask > 0] = 1.
            mask[mask < 0] = np.nan
            GeoData.create_dataset('nLw_' + str(band), (rows, columns), dtype='f', data=nLw[:, :, k] * mask)
            Ltemp[:, :, k][Ltemp[:, :, k] >= 1] = np.nan
            GeoData.create_dataset('Rrs_' + str(band), (rows, columns), dtype='f', data=Rrs[:, :, k] * mask)
            GeoData.create_dataset('Lrc_' + str(band), (rows, columns), dtype='f', data=Lrc[:, :, k])
            # GeoData.create_dataset('t_sen_' + str(band), (rows, columns), dtype='f', data=t_sensor[:, :, k])
            # GeoData.create_dataset('t_sol_' + str(band), (rows, columns), dtype='f', data=t_solar[:, :, k])
            # GeoData.create_dataset('tg_sol_' + str(band), (rows, columns), dtype='f', data=tg_sol[:, :, k])
            # GeoData.create_dataset('tg_sen_' + str(band), (rows, columns), dtype='f', data=tg_sen[:, :, k])

        # 参数文件，用于交叉定标
        GeoData.create_dataset('aer_model_min', (rows, columns), dtype='f', data=aer2[:, :, 0])
        GeoData.create_dataset('aer_model_max', (rows, columns), dtype='f', data=aer2[:, :, 1])
        GeoData.create_dataset('aer_model_ratio', (rows, columns), dtype='f', data=aer1[:, :, 0])
        for k, band in enumerate(bands):
            GeoData.create_dataset('aot_' + str(band), (rows, columns), dtype='f', data=taua[:, :, k])

        para_name = ['ozone', 'no2_strat', 'no2_tropo', 'pressure', 'windspeed', 'windangle', 'humidity']
        for k, nav in enumerate([o3, strat_no2, trop_no2, pressure, winds_peed, winddirection, rh]):
            GeoData.create_dataset(para_name[k], (rows, columns), dtype='f', data=nav)

        NavData = f_new.create_group("navigation_data")
        para_name = ['solz', 'senz', 'sola', 'sena', 'latitude', 'longitude']
        for k, nav in enumerate([sza, vza, saa, vaa, lat, lon]):
            NavData.create_dataset(para_name[k], (rows, columns), dtype='f', data=nav)

        f_new.close()
        endtime = datetime.datetime.now()
        print('processing time: ' + str((endtime - starttime).seconds) + ' seconds')