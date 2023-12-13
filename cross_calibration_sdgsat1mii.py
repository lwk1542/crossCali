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
from sharepy import get_filelist, read_aerosol_seadas, imagery_time, mask, spatial_limit
from l2gen import aerosol_radV2, atmosphericParameter, gas_transmittance, getglint, rayleigh_rad, whitecap_rad


class CrossCalibration(object):
    def __init__(self):
        self.sensorID = "sdgsat1_mii"
        self.sensorID_ref = "s3b_olci"  # s3b_olci,terra_modis,s3a_olci
        self.sensorID_tar_bands = [0, 1, 2, 3, 4, 5, 6]
        self.sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]

        self.sensor_alt = 550
        try:
            "如果每个文件的空间范围不一样"
            file = "G:\SDGsat\calibration\sea" + os.sep + "spatialLimit.txt"
            self.spa_lim = spatial_limit(txtfile=file)
            self.spatial_limit_index = 1
            print("spatial_limit_index: uncertain areas")
        except:
            "如果是一个固定的空间范围"
            self.spatial_limit_index = 0
            self.south, self.north, self.west, self.east = 13.5, 14.5, 110.867, 111.88  # 17.5, 19, 118, 120.5  # 21, 22, 125, 126 # 14, 20, 116, 120

        self.outdir=r"G:\SDGsat\calibration\sea\202303\result"
        path = r'G:\SDGsat\calibration\sea\202303'
        self.ref_filedir = path + os.sep + 'reference'  # 数据路径  idx, *args, path=None
        self.tar_filedir = path + os.sep + 'target'
        self.files = get_filelist('S3B', '_OL_1_EFR', '_seadas_rrs.hdf', path=self.ref_filedir, mode='all')

    def run_main(self):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        logfile = open(self.tar_filedir + os.sep + "crosscalibration_logfile.txt", "w+")
        logfile.write("target images directory:{}".format(self.tar_filedir))
        logfile.write("\n" + "reference images directory:{}".format(self.ref_filedir))
        logfile.write("\n" + "sensor_altitude:{}".format(self.sensor_alt))
        filepath_temp = None
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for num, ref_file in enumerate(self.files):
            # 匹配参考文件和待定标文件
            print('Processing:' + str(num + 1) + '/' + str(self.files.__len__()) + ' image')
            logfile.write("\n" + "=======第{}个参考文件：开始=============".format(num))
            logfile.write("\n" + "reference file:{}".format(os.path.basename(ref_file)))
            if any([self.sensorID_ref == 's3a_olci', self.sensorID_ref == 's3b_olci']):
                year, month, day, doy, date_str2 = imagery_time.obtain_time(sensor=self.sensorID_ref, file=ref_file)
            elif any([self.sensorID_ref == 'terra_modis', self.sensorID_ref == 'aqua_modis']):
                year, month, day, doy, date_str2 = imagery_time.obtain_time(sensor=self.sensorID_ref, file=ref_file)
            else:
                continue
            if self.sensorID == "sdgsat1_mii":
                from sdgsat1mii import read_img
                from sdgsat1mii import extra_info

            filepaths, self.lut_path, self.nirs_num, self.nirl_num = extra_info(tar_filedir=self.tar_filedir, date_str2=date_str2)
            for num_j, filepath in enumerate(filepaths):
                logfile.write("\n" + "Starting one new simulation")
                logfile.write("\n" + "=======第{}个目标文件：开始=============".format(num_j))
                logfile.write("\n" + "target file:{}".format(os.path.basename(filepath)))
                if self.spatial_limit_index == 1:
                    limit = self.spa_lim[os.path.basename(filepath)]
                    self.south, self.north, self.west, self.east = limit[1], limit[3], limit[0], limit[2]
                logfile.write(
                    "\n" + "south,north,west,east:{},{},{},{}".format(self.south, self.north, self.west, self.east))
                imagery_dict = read_img.Read(filepath=filepath, south=self.south, north=self.north, west=self.west,
                                             east=self.east, resize=0.3).run_main()  # 降分辨率
                imagery_dict["out_dir"] = self.outdir
                imagery_dict["tar_file"] = os.path.basename(filepath)
                imagery_dict["ref_file"] = os.path.basename(ref_file)
                imagery_dict["year"] = year
                imagery_dict["month"] = month
                imagery_dict["day"] = day
                imagery_dict["doy"] = doy
                result = self.simulate_ltoa(imagery_dict=imagery_dict, ref_file=ref_file)
                logfile = self.log_result(result, logfile)

        logfile.close()

    def simulate_ltoa(self, imagery_dict, ref_file):

        # 查找表路径
        rayleigh_lut_path = self.lut_path + os.sep + 'rayleigh'
        aerosol_lut_filepath = self.lut_path + os.sep + 'aerosol'
        doyi = int(imagery_dict["doy"])
        A, B, C, D, E = 1.00014, 0.01671, 0.9856002831, 3.452868, 360.
        fsol = ((A - B * np.cos(2. * np.pi * (C * doyi - D) / E) - 0.000014 * np.cos(
            4. * np.pi * (C * doyi - D) / E)) ** 2)

        starttime = datetime.datetime.now()

        # 读取参考传感器的气溶胶信息
        [Rrs, F0, delta, La_nirl, Fo_nirl, lat_ref, lon_ref, sza_ref, vza_ref, saa_ref, vaa_ref, \
         aermod_up_idx, aermod_low_idx] = read_aerosol_seadas.info(file=ref_file).run()
        rhoa_nirl = La_nirl * np.pi * fsol / (Fo_nirl * np.cos(sza_ref * np.pi / 180))

        # 取共同区域
        south_area = np.max([np.min(imagery_dict["lat"]), np.min(lat_ref)])
        north_area = np.min([np.max(imagery_dict["lat"]), np.max(lat_ref)])
        west_area = np.max([np.min(imagery_dict["lon"]), np.min(lon_ref)])
        east_area = np.min([np.max(imagery_dict["lon"]), np.max(lon_ref)])
        # 目标传感器
        loc1 = np.where((south_area < imagery_dict["lat"]) & (imagery_dict["lat"] < north_area) &
                        (west_area < imagery_dict["lon"]) & (imagery_dict["lon"] < east_area))
        if loc1[0].size < 10 or loc1[1].size < 10:
            return 0

        up, low, left, right = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])
        sza = imagery_dict["sza"][up:low, left:right]
        vza = imagery_dict["vza"][up:low, left:right]
        saa = imagery_dict["saa"][up:low, left:right]
        vaa = imagery_dict["vaa"][up:low, left:right]
        lat = imagery_dict["lat"][up:low, left:right]
        lon = imagery_dict["lon"][up:low, left:right]
        dn = imagery_dict["dn"][up:low, left:right, :]*1.
        judge = dn[:, :, 6] > 500
        dn[judge] = np.nan

        # 参考传感器
        loc2 = np.where(
            (south_area < lat_ref) & (lat_ref < north_area) & (west_area < lon_ref) & (lon_ref < east_area))
        if loc2[0].size < 10 or loc2[1].size < 10:
            return 0
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

        FoBAR = np.array(imagery_dict["F0"]) * fsol

        winds_peed, winddirection, pressure, o3, rh, water_vapor, strat_no2, trop_no2, taua = \
            atmosphericParameter.get(Lon=lon, Lat=lat, year=imagery_dict["year"], month=imagery_dict["month"],
                                     day=imagery_dict["day"], time='03:00')

        # 3. 瑞利光学厚度的气压校正
        # /* Pressure correct the Rayleigh optical thickness */
        # taur[ib] = l1rec->pr[ip] / p0 * l1file->Tau_r[ib]
        # Tau_r = rot.tau_r(bands=bands, ray_lut_path=rayleigh_lut_path)
        #
        factor = pressure / 1013.25
        taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * imagery_dict["Tau_r"].reshape(1, 1, -1)

        # 4. 臭氧透过率校正，臭氧从欧洲下载，臭氧消光截面不是从NASA下载，详情见函数内部。Mobely给的消光截面单位是错误的
        tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=sza, vza=vza, koz=imagery_dict["k_oz"],
                                                                       concentration=o3)
        tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=imagery_dict["k_no2"], sza=sza,
                                                                          vza=vza,
                                                                          strat_no2=strat_no2,
                                                                          trop_no2=trop_no2)
        tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=imagery_dict["t_co2"], sza=sza,
                                                                          vza=vza)
        tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=water_vapor, sza=sza,
                                                                          vza=vza,
                                                                          zia_table=imagery_dict["zia_table"])

        tg_sol = tg_solar_o3 * tg_solar_no2 * tg_sensor_co2 * tg_sensor_h2o  # 其它吸收暂时不考虑
        tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_solar_co2 * tg_solar_h2o

        # /* Correct for ozone absorption.  We correct for inbound and outbound here, then we put the inbound back
        # when computing Lw.*/ Ltemp[ib] = Ltemp[ib] / l1rec->tg_sol[ipb] / l1rec->tg_sen[ipb];
        # Ltemp = Lt / tg_sensor_o3 / tg_solar_o3  # 上下和下行透过率
        # Ltemp = Lt / tg_sen / tg_sol  # 上下和下行透过率

        # 卷云和极化校正：没做
        # / *Apply polarization correction * /

        # 5. 移除白帽反射

        rho_wc = whitecap_rad.calculate(U10=winds_peed, bands=imagery_dict["bands"])
        t_sen = np.empty(shape=(vza.shape[0], vza.shape[1], imagery_dict["bands"].size))
        t_sol = np.empty_like(t_sen)
        tLf = np.empty_like(t_sen)

        for i in range(imagery_dict["bands"].size):
            t_sen[:, :, i] = np.exp(-0.5 * (pressure / 1013.25) * taur[:, :, i] / np.cos(vza * np.pi / 180))
            t_sol[:, :, i] = np.exp(-0.5 * (pressure / 1013.25) * taur[:, :, i] / np.cos(sza * np.pi / 180))
            tLf[:, :, i] = rho_wc[:, :, i] * t_sol[:, :, i] * t_sen[:, :, i] * FoBAR[i] * np.cos(
                sza * np.pi / 180) / np.pi
        # Ltemp = Ltemp - tLf

        # 6. 移除瑞利贡献
        Lr = rayleigh_rad.rayleigh(rayleigh_lut_path=rayleigh_lut_path, sza=sza, vza=vza, saa=saa, vaa=vaa,
                                   F0=FoBAR, windspeed=winds_peed, pressure=pressure)

        airmass1 = 1 / np.cos(np.deg2rad(sza)) + 1 / np.cos(np.deg2rad(vza))
        # 7.氧气校正：只需要对750nm做，直接沿用了seawifs的校正系数
        a_o2 = gas_transmittance.oxygen_ray(airmass1)
        t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass1)
        Lr[:, :, self.nirs_num] = Lr[:, :, self.nirs_num] * 1.  # * a_o2  #
        scaleRayleigh = 1.0 - np.exp(-self.sensor_alt / 10)
        Lr = Lr * scaleRayleigh

        # 7.气溶胶贡献

        # 通过插值保证相同的地理位置,将参考传感器的观测几何信息插入到目标传感器的位置
        rhoa_nirl = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), rhoa_nirl.flatten(),
                                         (lat, lon), method='nearest')
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
                                                  bands=imagery_dict["bands"], nirl_num=self.nirl_num, sza_ref=sza_ref,
                                                  vza_ref=vza_ref, saa_ref=saa_ref, vaa_ref=vaa_ref, F0=FoBAR,
                                                  pressure=pressure, taur=imagery_dict["Tau_r"])
        if aerosol is None:
            return 1
        else:
            La, t_sensor, t_solar, taua = aerosol["La"], aerosol["t_sensor"], aerosol["t_solar"], aerosol["taua"]

        # 7. 耀斑反射
        wd_rad = np.deg2rad(winddirection)
        TLg = getglint.main_exec(sza=sza, vza=vza, vaa=vaa, saa=saa, taur=taur, La=La, F0=FoBAR,
                                 windspeed=winds_peed, winddirection=wd_rad, taua=taua, iter_num=1, mode=2)

        # 8. 离水辐射
        Rrs = Rrs[:, :, self.sensorID_ref_bands]
        Rrs_tar = np.zeros(shape=(sza.shape[0], sza.shape[1], Rrs.shape[2]))
        for j in range(Rrs.shape[2]):
            rrs = Rrs[up2: low2, left2: right2, j]
            Rrs_tar[:, :, j] = interpolate.griddata((lat_ref.flatten(), lon_ref.flatten()), rrs.flatten(),
                                                    (lat, lon), method='linear')
        FoBAR = FoBAR.reshape(1, 1, -1)
        nLw = Rrs_tar * FoBAR
        csza = np.zeros(shape=(sza.shape[0], sza.shape[1], 1))
        csza[:, :, 0] = np.cos(sza * np.pi / 180)
        Lw = nLw * t_solar * tg_sol * csza / fsol
        tLw = Lw / tg_sol * t_sensor

        # LTOA
        Lt_simu = tLf + Lr + TLg + La / fsol + tLw

        outfile = imagery_dict["out_dir"] + os.sep + imagery_dict["tar_file"][0:31] + "_" + \
                  os.path.splitext(imagery_dict["ref_file"])[0][0:13]+\
                  os.path.splitext(imagery_dict["ref_file"])[0][63:94]+"_crossCalibration.h5"
        f_new = h5py.File(outfile, 'a')
        f_new.attrs.create('target file', imagery_dict["tar_file"], shape=(1,), dtype='S80')
        f_new.attrs.create('reference file', imagery_dict["ref_file"], shape=(1,), dtype='S80')
        f_new.attrs.create('product time', starttime.strftime("%m/%d/%Y, %H:%M:%S"), shape=(1,), dtype='S80')
        f_new.attrs.create('author', 'Li Wenkai', shape=(1,), dtype='S30')
        f_new.attrs.create('email', 'lwk1542@scsio.ac.cn', shape=(1,), dtype='S26')
        f_new.attrs.create('method', 'Gordon; Wang; Hu;', shape=(1,), dtype='S26')

        (rows, columns) = dn[:, :, 0].shape
        GeoData = f_new.create_group("Geophysical Data")
        for k, band in enumerate(imagery_dict["bands"]):
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
        geomask = mask(sza=sza, saa=saa, vza=vza, vaa=vaa, sza_ref=sza_ref, saa_ref=saa_ref, vza_ref=vza_ref,
                       vaa_ref=vaa_ref)
        mask_ = f_new.create_group("Mask")
        mask_.create_dataset("geo_mask", (rows, columns), dtype='f', data=geomask)
        f_new.close()

    # def write_hdf(self):

    def log_result(self,result, log_obj):
        if result == 0:
            log_obj.write("\n" + "no matched pixel")
        elif result == 1:
            log_obj.write("\n" + "failed computing aerosol")
        else:
            log_obj.write("\n" + "finished simulating Ltoa")
        return log_obj


if __name__ == '__main__':
    CrossCalibration().run_main()
