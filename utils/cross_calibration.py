# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: cc_main.py
@time: 2021/4/1 13:37
@desc: cross calibration
"""
import datetime
import os

import h5py
import numpy as np
from scipy import interpolate

from atmospheric_correction.oceancolor_acnirv2.sharepy import readfile
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import aerosol_radV2, atmosphericParameter, \
    gas_transmittance
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import whitecap_rad, general, rayleigh_rad, \
    getglint


def mask(sza=None, saa=None, vza=None, vaa=None, sza_ref=None, saa_ref=None, vza_ref=None, vaa_ref=None):
    """
    对散射角、观测几何做限定
    """
    reaa = saa - vaa
    reaa = np.abs(reaa)
    reaa[reaa > 180.] = reaa[reaa > 180.] - 180
    temp = np.sqrt((1. - np.cos(vza * np.pi / 180.) ** 2) * (1. - np.cos(sza * np.pi / 180.) ** 2)) * np.cos(
        reaa * np.pi / 180.)

    temp_1 = -np.cos(vza * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_1[temp_1 < -1.] = -1.
    scatt1 = np.arccos(temp_1) * 180 / np.pi
    temp_2 = np.cos(vza * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_2[temp_2 > 1.] = 1.
    scatt2 = np.arccos(temp_2) * 180 / np.pi

    reaa_ref = saa_ref - vaa_ref
    reaa_ref = np.abs(reaa_ref)
    reaa_ref[reaa_ref > 180.] = reaa_ref[reaa_ref > 180.] - 180
    temp = np.sqrt((1. - np.cos(vza_ref * np.pi / 180.) ** 2) * (1. - np.cos(sza * np.pi / 180.) ** 2)) * np.cos(
        reaa_ref * np.pi / 180.)

    temp_1 = -np.cos(vza_ref * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_1[temp_1 < -1.] = -1.
    scatt1_ref = np.arccos(temp_1) * 180 / np.pi
    temp_2 = np.cos(vza_ref * np.pi / 180) * np.cos(sza * np.pi / 180) + temp
    temp_2[temp_2 > 1.] = 1.
    scatt2_ref = np.arccos(temp_2) * 180 / np.pi

    mask1 = scatt1 - scatt1_ref
    mask1[np.isnan(mask1)] = 999
    mask1[mask1 > 5] = np.nan
    mask1 = mask1 / mask1

    mask2 = scatt2 - scatt2_ref
    mask2[np.isnan(mask2)] = 999
    mask2[mask2 > 5] = np.nan
    mask2 = mask2 / mask2

    sza_mask3 = sza - sza_ref
    sza_mask3[np.isnan(sza_mask3)] = 999
    sza_mask3[sza_mask3 > 5] = np.nan
    sza_mask3 = sza_mask3 / sza_mask3

    sza[np.isnan(sza)] = 999
    sza[sza > 30] = np.nan
    sza_mask1 = sza / sza

    vza[np.isnan(vza)] = 999
    vza[vza > 30] = np.nan
    sza_mask2 = vza / vza
    mask = mask1 * mask2 * sza_mask1 * sza_mask2 * sza_mask3

    return mask


if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    south, north, west, east = 25, 40, 118, 128  # 17.5, 19, 118, 120.5  # 21, 22, 125, 126 # 14, 20, 116, 120
    sensor_alt = 780
    ref_filedir = r'G:\high_quality\manual_25_40_118_128'  # 数据路径
    tar_filedir = r'G:\high_quality\test_reaa_abs\L1B'
    files = general.get_filelist(ref_filedir, 'MOD.1KM.L2A.', '.hdf')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for num, ref_file in enumerate(files):
        # 匹配参考文件和待定标文件
        print('Processing:' + str(num + 1) + '/' + str(files.__len__()) + ' image')
        ref_sensorID = os.path.basename(ref_file)[0:3]
        if any([ref_sensorID == 'MOD', ref_sensorID == 'MYD']):
            date_str = os.path.basename(ref_file)[13:20]  # [10:17] MOD.1KM.L2A.A2008065.0255.061.2017255040540.hdf
            year, doy = date_str[0:4], date_str[4:7]
            date = datetime.datetime.strptime(year + doy, '%Y%j')
            date_str2 = date.strftime('%Y%m%d')
            month, day = date_str2[4:6], date_str2[6:8]
        else:
            continue

        target_files = general.get_filelistv2('H1B', date_str2, '_10.h5', path=tar_filedir, mode='all')
        for target_file in target_files:
            outfile = target_file[0: -3] + '_' + os.path.basename(ref_file)[0:-4] + '_crossCalibration.h5'
            if os.path.exists(outfile):
                continue
            if target_file.__len__() == 0:
                continue

            starttime = datetime.datetime.now()
            # 1. 读取待定标文件

            # 根据文件名确定查找表路径
            sensorID = os.path.basename(target_file)[0:3]
            print('processing: ' + sensorID)
            if sensorID == 'H1C':
                lut_path = r'LUT/HY1C_COCTS_LUTs'
            elif sensorID == 'H1A':
                lut_path = r'LUT/HY1A_COCTS_LUTs'
            elif sensorID == 'H1B':
                lut_path = r'LUT/HY1B_COCTS_LUTs'
            else:
                print('unidentified satellite sensor: ' + sensorID)
                continue
            rayleigh_lut_path = lut_path + os.sep + 'rayleigh'
            aerosol_lut_filepath = lut_path + os.sep + 'aerosol'

            # 1. 确定观测几何和地理位置
            #   备注：经过检查 H1C的替代定标系数是错的
            #   根据文件名选择读取程序以及相关设置
            if any([sensorID == 'H1A', sensorID == 'H1B', sensorID == 'H1C']):
                print('target file: ' + os.path.basename(target_file))
                out1 = readfile.hy1abc_l1ab(infile=target_file, north=north, south=south, west=west, east=east)
                if out1.__len__() == 0:
                    print('no expected file, executing next file')
                    continue
                sza, vza, saa, vaa, lat, lon, dn, bands, Fo, Tau_r, koz, kno2, tco2, zia_table, aw, bbw \
                    = out1

                date_str = os.path.basename(target_file)[17:25]
                year, month, day = date_str[0:4], date_str[4:6], date_str[6:8]
                date = datetime.datetime.strptime(year + month + day, '%Y%m%d')
                doy = date.strftime('%j')
                # 根据卫星传感器指定两个用于气溶胶估算的近红外波段，起始位0
                nirs_num = 6
                nirl_num = 7
            else:
                continue
            if out1 is None:
                continue

            doyi = int(doy)
            A, B, C, D, E = 1.00014, 0.01671, 0.9856002831, 3.452868, 360.
            fsol = ((A - B * np.cos(2. * np.pi * (C * doyi - D) / E) - 0.000014 * np.cos(
                4. * np.pi * (C * doyi - D) / E)) ** 2)
            # 读取参考传感器的气溶胶信息
            print('reference file: ' + os.path.basename(ref_file))

            Rrs, F0, delta, La_nirl, Fo_nirl, lat_ref, lon_ref, sza_ref, vza_ref, saa_ref, vaa_ref, \
            aermod_up_idx, aermod_low_idx = readfile.read_aerosol_info_seadas(infile=ref_file)
            rhoa_nirl = La_nirl * np.pi * fsol / (Fo_nirl * np.cos(sza_ref * np.pi / 180))

            # 取共同区域
            south_area = np.max([np.min(lat), np.min(lat_ref)])
            north_area = np.min([np.max(lat), np.max(lat_ref)])
            west_area = np.max([np.min(lon), np.min(lon_ref)])
            east_area = np.min([np.max(lon), np.max(lon_ref)])

            # 目标传感器
            loc1 = np.where((south_area < lat) & (lat < north_area) & (west_area < lon) & (lon < east_area))
            if loc1[0].size < 10 or loc1[1].size < 10:
                continue
            up, low, left, right = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])
            sza = sza[up:low, left:right]
            vza = vza[up:low, left:right]
            saa = saa[up:low, left:right]
            vaa = vaa[up:low, left:right]
            lat = lat[up:low, left:right]
            lon = lon[up:low, left:right]
            dn = dn[up:low, left:right, :]

            # 参考传感器
            loc2 = np.where(
                (south_area < lat_ref) & (lat_ref < north_area) & (west_area < lon_ref) & (lon_ref < east_area))
            if loc2[0].size < 10 or loc2[1].size < 10:
                continue
            up2, low2, left2, right2 = np.min(loc2[0]), np.max(loc2[0]), np.min(loc2[1]), np.max(loc2[1])
            lat_ref = lat_ref[up2:low2, left2:right2]
            lon_ref = lon_ref[up2:low2, left2:right2]
            aermod_up_idx = aermod_up_idx[up2:low2, left2:right2]
            aermod_low_idx = aermod_low_idx[up2:low2, left2:right2]
            delta = delta[up2:low2, left2:right2]
            La_nirl = La_nirl[up2:low2, left2:right2]

            sza_ref = sza_ref[up2:low2, left2:right2]
            vza_ref = vza_ref[up2:low2, left2:right2]
            saa_ref = saa_ref[up2:low2, left2:right2]
            vaa_ref = vaa_ref[up2:low2, left2:right2]
            rhoa_nirl = rhoa_nirl[up2:low2, left2:right2]

            vza[vza > 88] = 88
            vza[vza < 0] = 0
            sza[sza > 88] = 88
            sza[sza < 0] = 0
            # 2. 下载气象数据，根据海洋卫星经纬度插值出相应的气象参数：风速、气压、 臭氧，其它如水汽柱也可输出

            FoBAR = Fo * fsol

            winds_peed, winddirection, pressure, o3, rh, water_vapor, strat_no2, trop_no2, taua = \
                atmosphericParameter.get(Lon=lon, Lat=lat, year=year, month=month, day=day, time='03:00')

            # 3. 瑞利光学厚度的气压校正
            # /* Pressure correct the Rayleigh optical thickness */
            # taur[ib] = l1rec->pr[ip] / p0 * l1file->Tau_r[ib]
            # Tau_r = rot.tau_r(bands=bands, ray_lut_path=rayleigh_lut_path)
            #
            factor = pressure / 1013.25
            taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * Tau_r.reshape(1, 1, -1)

            # 4. 臭氧透过率校正，臭氧从欧洲下载，臭氧消光截面不是从NASA下载，详情见函数内部。Mobely给的消光截面单位是错误的
            tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=sza, vza=vza, koz=koz, concentration=o3)
            tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=kno2, sza=sza, vza=vza,
                                                                              strat_no2=strat_no2, trop_no2=trop_no2)
            tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=tco2, sza=sza, vza=vza)
            tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=water_vapor, sza=sza, vza=vza,
                                                                              zia_table=zia_table)

            tg_sol = tg_solar_o3 * tg_solar_no2 * tg_sensor_co2 * tg_sensor_h2o  # 其它吸收暂时不考虑
            tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_solar_co2 * tg_solar_h2o

            # /* Correct for ozone absorption.  We correct for inbound and outbound here, then we put the inbound back
            # when computing Lw.*/ Ltemp[ib] = Ltemp[ib] / l1rec->tg_sol[ipb] / l1rec->tg_sen[ipb];
            # Ltemp = Lt / tg_sensor_o3 / tg_solar_o3  # 上下和下行透过率
            # Ltemp = Lt / tg_sen / tg_sol  # 上下和下行透过率

            # 卷云和极化校正：没做
            # / *Apply polarization correction * /

            # 5. 移除白帽反射
            rho_wc = whitecap_rad.calculate(U10=winds_peed, bands=bands)
            t_sen = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))
            t_sol = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))
            tLf = np.empty(shape=(vza.shape[0], vza.shape[1], bands.size))

            for i in range(bands.size):
                t_sen[:, :, i] = np.exp(-0.5 * (pressure / 1013.25) * taur[:, :, i] / np.cos(vza * np.pi / 180))
                t_sol[:, :, i] = np.exp(-0.5 * (pressure / 1013.25) * taur[:, :, i] / np.cos(sza * np.pi / 180))
                tLf[:, :, i] = rho_wc[:, :, i] * t_sol[:, :, i] * t_sen[:, :, i] * FoBAR[i] * np.cos(
                    sza * np.pi / 180) / np.pi
            # Ltemp = Ltemp - tLf

            # 6. 移除瑞利贡献
            Lr = rayleigh_rad.rayleigh(rayleigh_lut_path=rayleigh_lut_path, sza=sza, vza=vza, saa=saa, vaa=vaa,
                                       F0=FoBAR, windspeed=winds_peed, pressure=pressure, sensorID=sensorID)
            airmass1 = 1 / np.cos(sza * np.pi / 180) + 1 / np.cos(vza * np.pi / 180)
            # 7.氧气校正：只需要对750nm做，直接沿用了seawifs的校正系数
            a_o2 = gas_transmittance.oxygen_ray(airmass1)
            t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass1)
            Lr[:, :, nirs_num] = Lr[:, :, nirs_num] * 1.  # * a_o2  #
            scaleRayleigh = 1.0 - np.exp(-sensor_alt / 10)
            Lr = Lr * scaleRayleigh

            # 7.气溶胶贡献

            # 通过插值保证相同的地理位置,将参考传感器的观测几何信息插入到目标传感器的位置
            rhoa_nirl = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), rhoa_nirl.flatten(), (lat, lon),
                                             method='nearest')

            # phase1 = interpolate.interpn(
            #         (aer_models[0, i]['wave_lut'].reshape(-1), aer_models[0, i]['scatt_lut'].reshape(-1)),
            #         aer_models[0, i]['phase_lut'],
            #         np.stack([[[band]], [[scatte_angle_]]], axis=2), method='linear', fill_value=None,
            #         bounds_error=False)

            delta = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), delta.flatten(), (lat, lon),
                                         method='nearest')
            aermod_up_idx = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), aermod_up_idx.flatten(),
                                                 (lat, lon), method='nearest')
            aermod_low_idx = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), aermod_low_idx.flatten(),
                                                  (lat, lon), method='nearest')

            sza_ref = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), sza_ref.flatten(), (lat, lon),
                                           method='linear')
            vza_ref = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), vza_ref.flatten(), (lat, lon),
                                           method='linear')
            saa_ref = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), saa_ref.flatten(), (lat, lon),
                                           method='linear')
            vaa_ref = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), vaa_ref.flatten(), (lat, lon),
                                           method='linear')
            # geomask=mask(sza=sza, saa=saa, vza=vza, vaa=vaa, sza_ref=sza_ref, saa_ref=saa_ref, vza_ref=vza_ref, vaa_ref=vaa_ref)
            #
            # sza_ref =sza_ref *geomask
            # vza_ref =vza_ref *geomask
            # saa_ref =saa_ref *geomask
            # vaa_ref =vaa_ref *geomask

            vza_ref[np.isnan(vza_ref)] = 80
            sza_ref[np.isnan(sza_ref)] = 80
            vza_ref[vza_ref > 80] = 80
            vza_ref[vza_ref < 0] = 0
            sza_ref[sza_ref > 80] = 80
            sza_ref[sza_ref < 0] = 0

            # 以上项无需依赖参考传感器的气溶胶信息，以下项则要依赖

            # 8.气溶胶反射
            aerosol = aerosol_radV2.cross_calibration(delta=delta, rhoa_nirl=rhoa_nirl, sza=sza, vza=vza,
                                                      saa=saa, vaa=vaa, aer_model_max=aermod_up_idx,
                                                      aer_model_min=aermod_low_idx,
                                                      aerosol_lut_filepath=aerosol_lut_filepath,
                                                      bands=bands, nirl_num=nirl_num, sza_ref=sza_ref,
                                                      vza_ref=vza_ref, saa_ref=saa_ref, vaa_ref=vaa_ref, F0=FoBAR,
                                                      pressure=pressure, taur=Tau_r)
            if aerosol is None:
                continue
            else:
                La, t_sensor, t_solar, taua = aerosol

            # 7. 耀斑反射
            wd_rad = winddirection * np.pi / 180
            TLg = getglint.main_exec(sza=sza, vza=vza, vaa=vaa, saa=saa, taur=taur, La=La, F0=FoBAR,
                                     windspeed=winds_peed, winddirection=wd_rad, taua=taua, iter_num=1, mode=2)

            # MODIS对COCTS
            Rrs = Rrs[:, :, [0, 1, 3, 4, 5, 8, 10, 12]]
            Rrs_tar = np.zeros(shape=(sza.shape[0], sza.shape[1], Rrs.shape[2]))
            for j in range(Rrs.shape[2]):
                rrs = Rrs[up2: low2, left2: right2, j]
                Rrs_tar[:, :, j] = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), rrs.flatten(),
                                                        (lat, lon), method='linear')
            # np.array([412, 443, 469, 488, 531, 551, 555, 645, 667, 678, 748, 859, 869, 1240, 1640, 2130])
            FoBAR = FoBAR.reshape(1, 1, -1)
            nLw = Rrs_tar * FoBAR
            csza = np.zeros(shape=(sza.shape[0], sza.shape[1], 1))
            csza[:, :, 0] = np.cos(sza * np.pi / 180)
            Lw = nLw * t_solar * tg_sol * csza / fsol
            tLw = Lw / tg_sol * t_sensor

            # LTOA
            Lt_simu = tLf + Lr + TLg + La / fsol + tLw

            f_new = h5py.File(outfile, 'a')
            f_new.attrs.create('target file', os.path.basename(target_file), shape=(1,), dtype='S80')
            f_new.attrs.create('reference file', os.path.basename(ref_file), shape=(1,), dtype='S80')
            f_new.attrs.create('product time', starttime.strftime("%m/%d/%Y, %H:%M:%S"), shape=(1,), dtype='S80')
            f_new.attrs.create('author', 'Li Wenkai, Tian liqiao', shape=(1,), dtype='S30')
            f_new.attrs.create('email', 'lwk1542@Hotmail.com', shape=(1,), dtype='S26')
            f_new.attrs.create('method', 'Gordon; Wang; Hu;', shape=(1,), dtype='S26')

            (rows, columns) = dn[:, :, 0].shape
            GeoData = f_new.create_group("Geophysical Data")
            for k, band in enumerate(bands):
                GeoData.create_dataset('tLf_' + str(band), (rows, columns), dtype='f', data=tLf[:, :, k])
                GeoData.create_dataset('TLg_' + str(band), (rows, columns), dtype='f', data=TLg[:, :, k])
                GeoData.create_dataset('La_' + str(band), (rows, columns), dtype='f', data=La[:, :, k])
                GeoData.create_dataset('Lr_' + str(band), (rows, columns), dtype='f', data=Lr[:, :, k])
                GeoData.create_dataset('Lw_' + str(band), (rows, columns), dtype='f', data=Lw[:, :, k])
                GeoData.create_dataset('t_sen_' + str(band), (rows, columns), dtype='f', data=t_sensor[:, :, k])
                GeoData.create_dataset('t_sol_' + str(band), (rows, columns), dtype='f', data=t_solar[:, :, k])
                GeoData.create_dataset('tg_sol_' + str(band), (rows, columns), dtype='f', data=tg_sol[:, :, k])
                GeoData.create_dataset('tg_sen_' + str(band), (rows, columns), dtype='f', data=tg_sen[:, :, k])
                GeoData.create_dataset('Lt_simu_' + str(band), (rows, columns), dtype='f', data=Lt_simu[:, :, k])
                GeoData.create_dataset('DN_' + str(band), (rows, columns), dtype='f', data=dn[:, :, k])
            GeoData.create_dataset('sza_targetsensor', (rows, columns), dtype='f', data=sza)
            GeoData.create_dataset('vza_targetsensor', (rows, columns), dtype='f', data=vza)
            GeoData.create_dataset('saa_targetsensor', (rows, columns), dtype='f', data=saa)
            GeoData.create_dataset('vaa_targetsensor', (rows, columns), dtype='f', data=vaa)
            GeoData.create_dataset('sza_referencesensor', (rows, columns), dtype='f', data=sza_ref)
            GeoData.create_dataset('vza_referencesensor', (rows, columns), dtype='f', data=vza_ref)
            GeoData.create_dataset('saa_referencesensor', (rows, columns), dtype='f', data=saa_ref)
            GeoData.create_dataset('vaa_referencesensor', (rows, columns), dtype='f', data=vaa_ref)

            NavData = f_new.create_group("Navigation Data")
            para_name = ['lat', 'lon']
            for k, nav in enumerate([lat, lon]):
                NavData.create_dataset(para_name[k], (rows, columns), dtype='f', data=nav)
            f_new.close()
