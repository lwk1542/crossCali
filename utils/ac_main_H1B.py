# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: ac_main_H1B.py
@time: 2021/7/4 20:42
@desc:
"""
import datetime
import os

import h5py
import numpy as np
from scipy import ndimage

import predefine
from atmospheric_correction.oceancolor_acnirv2.sharepy import readfile
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import aerosol_radV2, atmosphericParameter, gas_transmittance, \
    get_rhown_nir
from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import whitecap_rad, general, get_chl, \
    rayleigh_rad, getglint, brdf as brdfmodel

if __name__ == '__main__':

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    south, north, west, east = 15, 24.22, 110, 121.5  # 21, 22, 125, 126 # 14, 20, 116, 120
    sensor_alt = 780
    nonzeroNIR = "want_nirLw"
    filepath = r'G:\high_quality\H1B_70images\crosscalibration\select'  # 数据路径
    files = general.get_filelistv2('H1B_RICH_OCT_L1B', "_crossCalibration.h5", "MOD.1KM.", path=filepath, mode='all')
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    for num, infile in enumerate(files):
        print('Processing:' + str(num + 1) + '/' + str(files.__len__()) + ' image: ' + os.path.basename(infile))
        outfile = infile[0: -3] + '_L2A_WATERS_NIR-AC_v1.h5'
        if os.path.exists(outfile):
            if os.path.getsize(outfile) / 1024. / 1024 > 10:
                continue
            else:
                os.remove(outfile)
        else:
            pass

        # 根据文件名确定查找表路径
        sensorID = os.path.basename(infile)[0:3]
        print('sensor ID: ' + sensorID)
        if sensorID == 'H1A':
            lut_path = r'LUT/HY1A_COCTS_LUTs'
        elif sensorID == 'H1B':
            lut_path = r'LUT/HY1B_COCTS_LUTs'
        elif sensorID == 'H1C':
            lut_path = r'LUT/HY1C_COCTS_LUTs'
        elif sensorID == 'MOD':
            lut_path = r'LUT/Terra_modis_LUTs'
        elif sensorID == 'MYD':
            lut_path = r'LUT/Aqua_modis_LUTs'
        else:
            print('unidentified satellite sensor: ' + sensorID)
            continue
        print('look-up table directory: ' + lut_path)
        rayleigh_lut_path = lut_path + os.sep + 'rayleigh'
        aerosol_lut_filepath = lut_path + os.sep + 'aerosol'
        fqfile = r'LUT' + os.sep + 'morel_fq.h5'

        starttime = datetime.datetime.now()
        # 1. 确定观测几何和地理位置
        #   备注：经过检查 H1C的替代定标系数是错的
        #   根据文件名选择读取程序以及相关设置

        if any([sensorID == 'H1A', sensorID == 'H1B', sensorID == 'H1C']):
            out1 = readfile.hy1abc_l1ab(infile=infile, north=north, south=south, west=west, east=east)
            date_str = os.path.basename(infile)[17:25]
            year, month, day = date_str[0:4], date_str[4:6], date_str[6:8]
            # 根据卫星传感器指定两个用于气溶胶估算的近红外波段，起始位0
            num_443 = 1
            num_490 = 2
            num_520 = 3
            num_555 = 4
            num_670 = 5
            red = num_670
            nirs_num = 6
            nirl_num = 7
            nwvis = 6
        elif any([sensorID == 'MOD', sensorID == 'MYD']):
            out1 = readfile.modis_l1b(infile=infile, north=north, south=south, west=west, east=east)
            date_str = os.path.basename(infile)[10:17]
            year, doy = date_str[0:4], date_str[4:7]
            date = datetime.datetime.strptime(year + doy, '%Y%j')
            date_str2 = date.strftime('%Y%m%d')
            year, month, day = date_str2[0:4], date_str2[4:6], date_str2[6:8]
            if sensorID == 'MOD':
                num_443 = None
                num_490 = 2
                num_520 = 3
                num_555 = None
                num_670 = None
                red = num_670
                nirs_num = 10
                nirl_num = 12
                nwvis = None  # 可见光波段数量
            else:
                num_443 = None
                num_490 = 2
                num_520 = 3
                num_555 = None
                num_670 = None
                red = num_670
                nirs_num = None
                nirl_num = None
                nwvis = None  # 可见光波段数量
        elif all(idxi in os.path.basename(infile) for idxi in ['H1B', 'crossCalibration.h5']):
            out1 = readfile.simulated_radiance_hy1(infile=infile, north=north, south=south, west=west, east=east)
            date_str = os.path.basename(infile)[17:25]
            year, month, day = date_str[0:4], date_str[4:6], date_str[6:8]
            # 根据卫星传感器指定两个用于气溶胶估算的近红外波段，起始位0
            num_443 = 1
            num_490 = 2
            num_520 = 3
            num_555 = 4
            num_670 = 5
            red = num_670
            nirs_num = 6
            nirl_num = 7
            nwvis = None  # 可见光波段数量
        else:
            out1 = None
            year, month, day = None, None, None
            num_443 = None
            num_490 = 2
            num_520 = 3
            num_555 = None
            num_670 = None
            red = num_670
            nirs_num = None
            nirl_num = None
            nwvis = None  # 可见光波段数量

        if out1 is None:
            continue
        sza, vza, saa, vaa, lat, lon, Lt, bands, Fo, Tau_r, koz, kno2, tco2, zia_table, aw, bbw = out1

        vza[vza > 88] = 88
        vza[vza < 0] = 0
        sza[sza > 88] = 88
        sza[sza < 0] = 0
        # 2. 下载气象数据，根据海洋卫星经纬度插值出相应的气象参数：风速、气压、 臭氧，其它如水汽柱也可输出
        date = datetime.datetime.strptime(year + month + day, '%Y%m%d')
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
        tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=sza, vza=vza, koz=koz, concentration=o3)
        tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=kno2, sza=sza, vza=vza,
                                                                          strat_no2=strat_no2, trop_no2=trop_no2)
        tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=tco2, sza=sza, vza=vza)
        tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=water_vapor, sza=sza, vza=vza,
                                                                          zia_table=zia_table)

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
                                   F0=FoBAR, windspeed=winds_peed, pressure=pressure, sensorID=sensorID)
        airmass1 = 1 / np.cos(sza * np.pi / 180) + 1 / np.cos(vza * np.pi / 180)
        # 7.氧气校正：只需要对750nm做，直接沿用了seawifs的校正系数,仅需要对特定传感器校正
        a_o2 = gas_transmittance.oxygen_ray(airmass1)
        t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass1)
        lr[:, :, nirs_num] = lr[:, :, nirs_num]  # / a_o2  #
        scaleRayleigh = 1.0 - np.exp(-sensor_alt / 10)
        lr = lr * scaleRayleigh
        Lrc = (Ltemp - lr)
        Rrc = np.pi*Lrc/Fo_/np.cos(np.pi*sza.reshape(sza.shape[0],sza.shape[1],1)/180)
        Ltemp = Ltemp - lr

        # Ltemp[:, :, nirs_num] = Ltemp[:, :, nirs_num] / t_o2

        # 8.非F/Q brdf校正：不做了

        # /* ------------------------------------------------------------------------------------------------------ */
        # /* Begin interations for aerosol with corrections for non-zero nLw(NIR)近红外波段不等于0 */
        # /* ------------------------------------------------------------------------------------------------------ */
        # / *Initialize tLw as surface + aerosol radiance * /
        # taua = 0.1

        chl = np.full(shape=Ltemp[:, :, -1].shape, fill_value=predefine.thresholds().seed_chl)
        if nonzeroNIR == "want_nirLw":
            last_tLw_nir = np.zeros_like(Ltemp)
            tLw_nir = np.zeros_like(Ltemp)
            Rrs = np.zeros_like(Ltemp)
            Rrs[:, :, num_555] = predefine.thresholds().seed_green
            Rrs[:, :, num_670] = predefine.thresholds().seed_red
        else:
            # 给其它方法参数定义预留的位置
            last_tLw_nir = None
            tLw_nir = None
            Rrs = None
            Rrs[:, :, num_555] = None
            Rrs[:, :, num_670] = None

        mu0 = np.cos(sza * np.pi / 180)
        mu0 = mu0.reshape(sza.shape[0], sza.shape[1], 1)
        brdf = np.ones_like(mu0)

        #  /* Initialize iteration loop */
        last_iter = np.zeros_like(sza)
        iter_num = 0                # 最大迭代次数
        iterx = np.zeros_like(sza)  # 每个像元的跌打次数
        # iterx = np.zeros_like(sza)  # 每个像元的跌打次数
        last_refl_nir = np.full(shape=sza.shape, fill_value=100.)
        iter_reset = np.zeros_like(sza)
        iter_max = np.full_like(sza, fill_value=predefine.thresholds().aer_iter_max)
        tLw_final = np.full_like(Ltemp, fill_value=np.nan)

        #     chl = seed_chl;
        #     iter = 0;
        #     last_iter = 0;
        #     iter_max = aer_iter_max;
        #     iter_min = aer_iter_min;
        #     iter_reset = 0;
        #     last_refl_nir = 100.;
        #     want_glintcorr = 0;

        cslp = 1. / (predefine.thresholds().ctop - predefine.thresholds().cbot)
        cint = -cslp * predefine.thresholds().cbot

        # while iter_num <= predefine.thresholds().glint_iter_max:
        # 海洋1b星对近红外波段做平滑
        if sensorID == "H1B":
            Ltemp[:, :, -2] = ndimage.uniform_filter(Ltemp[:, :, -2], size=8)
            Ltemp[:, :, -1] = ndimage.uniform_filter(Ltemp[:, :, -1], size=8)

        while last_iter.min() == 0:    # last_iter是一个数组，任何一个像元没有达到停止迭代条件，均需要继续迭代
            iterx = iterx + 1
            status = 0
            print('迭代计算耀斑和气溶胶贡献: 第 ' + str(iter_num + 1) + ' 次')
            # 9.
            # 耀斑福亮度估算, First, the measured Lt(λ) and the wind
            # M. Wang and S. Bailey, "Correction of sun glint contamination on the SeaWiFS ocean and atmosphere products,"
            # Appl. Opt. 40, 4790-4798 (2001)
            # 第一次其实只需要做出近红外波段的耀斑，助后面选出气溶胶模型
            # print("computing glint contribution")
            if iter_num == 0:
                mode = 2
            else:
                mode = 2
            wd_rad = winddirection * np.pi / 180

            # /* Initialize tLw as surface + aerosol radiance */
            tLw = Ltemp * 1.

            TLg = getglint.main_exec(iter_num=iter_num, sza=sza, vza=vza, vaa=vaa, saa=saa, taur=taur, La=tLw, F0=FoBAR,
                                     windspeed=winds_peed, winddirection=wd_rad, taua=taua, mode=mode)
            tLw = Ltemp - TLg

            # 10./* Adjust for non-zero NIR water-leaving radiances using IOP model */

            # 估算近红外波段的水体福亮度贡献,目前只有这一种方法
            # /* Adjust for non-zero NIR water-leaving radiances using IOP model */
            if nonzeroNIR == "want_nirLw":
                rhown_nir = get_rhown_nir.get_rhown_eval(num_443=num_443, num_555=num_555, num_670=num_670,
                                                         nirs_num=nirs_num,
                                                         nirl_num=nirl_num, Rrs=Rrs, chl=chl, aw=aw, bbw=bbw,
                                                         fqfile=fqfile,
                                                         bands=bands, sza=sza, vza=vza, saa=saa, vaa=vaa)
                for ib in range(nirs_num, nirl_num + 1):
                    tLw_nir[:, :, ib] = (rhown_nir[:, :, ib] / np.pi * Fo[ib] * mu0[:, :, 0] * t_sol[:, :, ib] *
                                         t_sen[:, :, ib] / brdf[:, :, 0])
                    #  /* Iteration damping */
                    tLw_nir[ib] = ((1.0 - predefine.thresholds().df) * tLw_nir[ib] + predefine.thresholds().df *
                                   last_tLw_nir[ib])

                    # /* Ramp-up ?*/
                    tLw_nir[:, :, ib][(0 < chl) & (chl < predefine.thresholds().cbot)] = 0.
                    loc_temp = (predefine.thresholds().cbot < chl) & (chl < predefine.thresholds().ctop)
                    tLw_nir[:, :, ib][loc_temp] = tLw_nir[:, :, ib][loc_temp] * (cslp * chl + cint)[loc_temp]
                    tLw[:, :, ib] = tLw[:, :, ib] - tLw_nir[:, :, ib]
                    del loc_temp
            else:
                tLw_nir=None

            # 11.气溶胶
            #  近红外波段的气溶胶福亮度贡献
            l_nir1 = tLw[:, :, nirs_num] - tLw_nir[:, :, nirs_num]
            l_nir2 = tLw[:, :, nirl_num] - tLw_nir[:, :, nirl_num]
            # if status == 0:
            print("computing aerosol radiance...")
            aero_out = aerosol_radV2.calculate(l_a_nir1=l_nir1, l_a_nir2=l_nir2, lon=lon, lat=lat, F0=FoBAR,
                                               bands=bands,
                                               aerosol_lut_filepath=aerosol_lut_filepath,
                                               sza=sza, saa=saa, vza=vza, vaa=vaa,
                                               winds_peed=winds_peed, pressure=pressure,
                                               relative_humidity=rh, nirl_num=nirl_num, nirs_num=nirs_num,
                                               taur=Tau_r)
            if aero_out is None:
                break
            La, t_sensor, t_solar, taua, aer1, aer2 = aero_out

            if aero_out is None:
                continue
            # if status == 0:
            # /* subtract aerosol and normalize */
            tLw = tLw - La
            Lw = tLw / t_sensor * tg_sol
            # nLw = Lw / t_solar / tg_sol / np.cos(sza*np.pi/180) / fsol * brdf[ib];

            nLw = Lw / t_solar / tg_sol / mu0 / fsol * brdf

            # /* Compute new estimated chlorophyll */
            # ***************************************************************
            if nonzeroNIR == "want_nirLw":
                refl_nir = Rrs[:,:,red]
                for ib in range(nirs_num, nirl_num + 1):
                    last_tLw_nir[:, :, ib] = tLw_nir[:, :, ib]
                del ib
                for ib in range(nwvis):
                    Rrs[:, :, ib] = nLw[:, :, ib] / FoBAR[ib]
            else:
                refl_nir = Rrs[:,:,red]
                pass

            chl = get_chl.get_default_chl(rrs=Rrs, bands=bands, b443=num_443, b490=num_490, b520=num_520, b555=num_555,
                                          b670=num_670)
            # // if we passed atmospheric correction but the spectral distribution of
            # // Rrs is bogus (chl failed), assume this is a turbid-water case and
            # // reseed iteration as if all 670 reflectance is from water.

            #  if (chl == badchl && iter_reset == 0 && iter < iter_max)
            loc_temp = ((chl == predefine.thresholds().chlbad) & (iter_reset == 0) & (iterx < iter_max))
            chl[loc_temp] = 10
            Rrs[:, :, red][loc_temp] = 1.0 * (Ltemp[:, :, red][loc_temp] - TLg[:, :, red][loc_temp]) / \
                                       t_sol[:, :, red][loc_temp] / tg_sol[:, :, red][loc_temp] / mu0[:, :, 0][loc_temp] / \
                                       FoBAR[red]
            iter_reset[loc_temp] = 1
            del loc_temp
            #  if (chl == badchl && iter_reset == 1 && iter < iter_max)
            loc_temp = ((chl == predefine.thresholds().chlbad) & (iter_reset == 1) & (iterx < iter_max))
            chl[loc_temp] = 10
            Rrs[:, :, red][loc_temp] = 1.0 * (Ltemp[:, :, red][loc_temp] - TLg[:, :, red][loc_temp]) / \
                                       t_sol[:, :, red][loc_temp] / tg_sol[:, :, red][loc_temp] / mu0[:, :, 0][
                                           loc_temp] / FoBAR[red]
            iterx[loc_temp] = iter_max[loc_temp]
            iter_reset[loc_temp] = 2
            del loc_temp

            # /* Shall we continue iterating */
            # 找出停止迭代的像元
            # 暂时假设结果为nlw

            tLw_final_temp = tLw * 1.

            # loc_temp=iterx > predefine.thresholds().aer_iter_max
            # last_iter[loc_temp]=1

            if iter_num > predefine.thresholds().aer_iter_max:
                last_iter = np.ones_like(tLw_final[:, :, 0])
                for ib in range(bands.size):
                    tLw_final[:, :, ib] = np.nanmean(np.dstack((tLw_final[:, :, ib], tLw_final_temp[:, :, ib])), axis=2)
            else:
                last_iter = np.zeros_like(sza)
                loc_temp = ((np.abs(refl_nir - last_refl_nir) < np.abs(predefine.thresholds().nir_chg * refl_nir)) | (
                            refl_nir < 0.0))
                last_iter[loc_temp] = 1
                tLw_final_temp[~loc_temp] = np.nan  # 这部分不在迭代
                tLw[loc_temp] = np.nan  # 这部分继续迭代计算
                for ib in range(bands.size):
                    tLw_final[:, :, ib] = np.nanmean(np.dstack((tLw_final[:, :, ib], tLw_final_temp[:, :, ib])), axis=2)

            last_refl_nir = refl_nir
            iter_num = iter_num + 1
        # 至此，迭代完成

        #  /* Compute f/Q correction and apply to nLw */
        brdf_mod = brdfmodel.BRDF(vza=vza, sza=sza, vaa=vaa, saa=saa, bands=bands, F0=FoBAR, chl=chl, nlw=nLw, b443=num_443,
                             b490=num_490, b520=num_520, b555=num_555, b670=num_670, foqopt="FOQMOREL", ws=winds_peed,
                             fqfile=fqfile)
        brdf = brdf_mod.ocbrdf()

        #  /* Compute final Rrs */
        Lw = tLw_final / t_sensor * tg_sol
        # nLw = Lw / t_solar / tg_sol / np.cos(sza*np.pi/180) / fsol * brdf[ib];
        csza = np.zeros(shape=(sza.shape[0], sza.shape[1], 1))
        csza[:, :, 0] = np.cos(sza * np.pi / 180)
        nLw = Lw / t_solar / tg_sol / csza / fsol * brdf

        Rrs = nLw / Fo_

        # /* Compute final chl from final nLw (needed for flagging) */
        chl = get_chl.get_default_chl(rrs=Rrs, bands=bands, b443=num_443, b490=num_490, b520=num_520, b555=num_555, b670=num_670)

        #   /*Determine Raman scattering contribution to Rrs*/
        # run_raman_cor(l2rec, ip);

        # 写入文件
        f_new = h5py.File(outfile, 'a')
        f_new.attrs.create('input file', os.path.basename(infile), shape=(1,), dtype='S80')
        f_new.attrs.create('product time', starttime.strftime("%m/%d/%Y, %H:%M:%S"), shape=(1,), dtype='S80')
        f_new.attrs.create('author', 'Li Wenkai', shape=(1,), dtype='S8')
        f_new.attrs.create('email', 'lwk1542@Hotmail.com', shape=(1,), dtype='S26')
        f_new.attrs.create('method', 'NIR atmospheric correction', shape=(1,), dtype='S26')
        (rows, columns) = Rrs[:, :, 0].shape
        GeoData = f_new.create_group("geophysical_data")
        for k, band in enumerate(bands):
            GeoData.create_dataset('Lt_' + str(band), (rows, columns), dtype='f', data=Lt[:, :, k])
            GeoData.create_dataset('Lr_' + str(band), (rows, columns), dtype='f', data=lr[:, :, k])
            GeoData.create_dataset('tLf_' + str(band), (rows, columns), dtype='f', data=tLf[:, :, k])
            GeoData.create_dataset('TLg_' + str(band), (rows, columns), dtype='f', data=TLg[:, :, k])
            GeoData.create_dataset('La_' + str(band), (rows, columns), dtype='f', data=La[:, :, k])
            mask = np.nan_to_num(nLw[:, :, k], nan=-999, posinf=-999, neginf=-999)
            mask[mask > 0] = 1.
            mask[mask < 0] = np.nan
            GeoData.create_dataset('nLw_' + str(band), (rows, columns), dtype='f', data=nLw[:, :, k] * mask)
            Rrs[:, :, k][Rrs[:, :, k] >= 1] = np.nan
            GeoData.create_dataset('Rrs_' + str(band), (rows, columns), dtype='f', data=Rrs[:, :, k] * mask)
            GeoData.create_dataset('Lrc_' + str(band), (rows, columns), dtype='f', data=Lrc[:, :, k])
            GeoData.create_dataset('t_sen_' + str(band), (rows, columns), dtype='f', data=t_sensor[:, :, k])
            GeoData.create_dataset('t_sol_' + str(band), (rows, columns), dtype='f', data=t_solar[:, :, k])
            GeoData.create_dataset('tg_sol_' + str(band), (rows, columns), dtype='f', data=tg_sol[:, :, k])
            GeoData.create_dataset('tg_sen_' + str(band), (rows, columns), dtype='f', data=tg_sen[:, :, k])

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