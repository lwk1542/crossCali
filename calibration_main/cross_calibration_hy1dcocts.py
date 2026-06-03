# -*- coding: utf-8 -*-
"""
@Time    : 2026/5/12 10:49
@FileName: cross_calibration_sdgsat1miiv3.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""

# from datetime import datetime, timedelta
# from scipy import interpolate
from pathlib import Path
import numpy as np
import datetime
from typing import Literal
# import glob
import h5py
import cv2
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from AC_l2gen.utils import scene_time, esdist
from AC_l2gen.sensor import read_img_info
from AC_l2gen.sharepy import read_aerosol_seadas, read_tif, interp, setup_logging
from AC_l2gen.l2gen import atmosphericParameter, gas_transmittance, aerosol_rad, mask_crossCalibration,\
    whitecap_rad, read_lut, getglint,  ray_rad_idl


import logging




class SimulationLtoa(object):
    def __init__(self,  target_file: str, ref_file: str, outfile: str,sensorID: Literal["hy1dcocts","hy1dcocts"],
                    sensorid_ref: Literal["sentinel3aolci", "sentinel3bolci","modisa"]):
        """
        Args:
        self.sensorID = "hy1dcocts"  #"sdgsat1mii"
        """
        # ++++++++++++++++++++++++++++++++++需要设置的参数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.target_file = target_file # 如果
        self.sensorID = sensorID  #"sdgsat1mii"     
        # self.sensorID_ref = sensorid_ref  # s3b_olci,terra_modis,s3a_olci
        self.sensor_alt = 550
        self.sensorID_ref_bands = None  
        self.block_size_rows = None  # 后面设置一次全部读取
        self.ref_file = Path(ref_file)
        self.outfile=Path(outfile)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.rh = None
        self.winds_peed = None
        self.Rrs = None
        self.Rrs_ref = None
        self.rhoa_nirl = None
        self.delta = None
        self.aermod_low_idx = None
        self.aermod_up_idx = None
        self.lon_ref = None
        self.lat_ref = None
        self.output_info = None
        self.mask = None
        self.Lt_simu = None
        self.tLw = None
        self.t_solar = None
        self.t_sensor = None
        self.TLg = None
        self.La = None
        self.Lr = None
        self.tLf = None
        self.taur = None
        self.ref_filedir = None
        self.outdir = None
        self.south = None
        self.north = None
        self.east = None
        self.west = None
        self.vaa_ref = None
        self.saa_ref = None
        self.vza_ref = None
        self.sza_ref = None
        self.file_index = None
        self.dtype = np.float64
        self.taua_upscale = None
        self.aerosol_lut_info = None
        self.rayleigh_lut_info = None
        self.sensor_info = None
        self.infile = None
        self.fsol = None
        self.Tau_r = None
        self.winddirection = None
        self.pressure = None
        self.taua = None
        self.Fo_ = None
        self.fqfile = None
        self.FoBAR = None
        self.lat = None
        self.lon = None
        self.sza = None
        self.vza = None
        self.saa = None
        self.vaa = None
        self.Lt = None
        self.t_sol = None
        self.t_sen = None
        self.tg_sen = None
        self.tg_sol = None
        self.block_num = None

    def run_main(self) -> str:
        """
        主函数：
        Returns:
        """
        # 一. 获取传感器相关的信息
        self.sensor_info, self.rayleigh_lut_info, self.aerosol_lut_info = read_lut.CommonVariable(
            sensor_id=self.sensorID).get()
        (self.bands, self.Fo, self.Tau_r, self.k_oz, self.t_co2, self.k_no2, self.Zia_table, self.awhite, self.aw,
         self.bbw, self.oobwv, self.ooblw, self.wed, self.waph) = self.sensor_info
        self.fqfile = "share" + os.sep + "common" + os.sep + 'morel_fq.h5'
        date_str = scene_time(self.target_file)
        self.pre_parameters()
        # 读取目标影像
        # 二. 获取影像相关的信息
        # self.block_size_rows = self.rows()  # 一次读取原始影像的行数
        self.get_img_info(self.target_file)
        # 读取参考影像
        self.get_refimg_info(infile=self.ref_file)
        self.output_info = {"tar_file": self.target_file, "ref_file": self.ref_file}

        for self.block_num, d_i_temp in enumerate(self.data_Iterator):  # 每个传感器的迭代器数据内容可能不一样，针对性处理
            (data, gains, offsets, self.lon, self.lat, self.vaa, self.vza, self.saa, self.sza) = d_i_temp
            # "读取的依然是一整个文件"
            self.Lt = data * gains.reshape(1, 1, -1) * 1. + offsets.reshape(1, 1, -1)
            # 取共同区域
            exec1 = self.study_area()
            if exec1 == 0:
                return 0
            if exec1 == 1:
                return 1
            (self.rows_org, self.columns_org) = self.Lt[:, :, 0].shape

            print("***********参考影像{}".format(os.path.basename(self.ref_file)))
            # 1. 气象数据加载/计算气体吸收
            self.taur, self.tg_sol, self.tg_sen = self.meteor_para_and_gas_absorb()
            logging.info("气象数据加载/计算气体吸收完成")
            
            # 3.瑞利
            self.Lr = self.rayleigh()
            logging.info("瑞利散射计算完成")

            # 2. 白帽反射
            self.tLf, self.t_sen, self.t_sol = self.whitecap()
            logging.info("白帽反射计算完成")

            # 4气溶胶
            self.La, self.t_sensor, self.t_solar, self.taua = self.aerosol()
            logging.info("气溶胶计算完成")
            if self.La is None:
                return []
            # 5. 耀斑反射
            self.TLg = self.glint()
            logging.info("耀斑反射计算完成")
            # 6. 离水
            self.tLw = self.water()
            logging.info("离水计算完成")
            # LTOA
            airmass = 1 / np.cos(np.deg2rad(self.sza)) + 1 / np.cos(np.deg2rad(self.vza))
            # 7.氧气校正：
            t_o2 = gas_transmittance.oxygen_aer(airmass)
            temp = self.TLg + self.La + self.tLw
            temp[:, :, self.nirs_num] = temp[:, :, self.nirs_num] / t_o2
            self.Lt_simu = (self.tLf + self.Lr + temp) * self.tg_sol * self.tg_sen
            logging.info("Lt计算完成")
            self.mask = self.geomask()
            logging.info("几何遮罩计算完成")

        outfile_name = self.output()
        logging.info("输出文件: {}".format(outfile_name))

        return outfile_name

    def pre_parameters(self):
        """
        需要手动调整的参数
        """
        # self.south, self.north, self.west, self.east = self.limit[1], self.limit[3], self.limit[0], self.limit[2]
        
        if isinstance(self.ref_file, Path):
            file_str = self.ref_file.name  # 转换为字符串
        else:
            file_str = os.path.basename(self.ref_file)
            
        if 'S3A' in file_str:
            self.sensorID_ref = "s3a_olci"
            self.sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
        if 'S3B' in file_str:
            self.sensorID_ref = "s3b_olci"
            self.sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
        if self.sensorID == "hy1dcocts" and 'MYD' in file_str:
            self.sensorID_ref = "modis"
            self.sensorID_ref_bands = [0, 1, 3, 4, 6, 8, 10, 12]
        # self.ref_filedir = self.father_dir + os.sep + 'reference'  # 数据路径

    def cloud_land_mask(self, lt):
        if self.sensorID == "sdgsat1mii":
            # from sensor.sdgsat1mii import mask
            # lt = mask.cloud_land_mask(lt=lt, sza=self.sza, F0=self.Fo_)
            (rows_, columns_, _) = lt.shape
            # mu = np.cos(np.deg2rad(sza)).reshape(rows_, columns_, 1)
            # rhot = np.pi * lt / F0 / mu
            m1 = lt[:, :, 5] > lt[:, :, 6]
            z = m1
            for i in range(7):
                lt[:, :, i][~z] = np.nan
                # lt[:, :, i][lt[:, :, i] > F0[0, 0, i]] = np.nan
            ret, binary = cv2.threshold(lt[:, :, 0], 0, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3))
            open1 = cv2.erode(binary, kernel, iterations=2)  # 腐蚀
            open1[np.isnan(open1)] = np.nan
            open1 = open1 / open1
            open1[open1 <= 0] = np.nan
            open1[open1 > 260] = np.nan
            open1[~np.isnan(open1)] = 1
            # lt = lt * open1.reshape(self.rows_chunk, self.columns_chunk, 1)
            return lt * open1.reshape(rows_, columns_, 1)
        return lt

    def rows(self):
        columns, rows_ = read_tif.GdalReadTif(in_file=self.target_file).get_info()
        return rows_

    def study_area(self):
        """
        计算目标传感器与参考传感器的共同研究区域，并对数据进行裁剪。
        已处理跨 180° 经度线的情况，并加入了几何条件筛选。
        """
        
        _lat_ref=self.ref_dic['latitude']
        _lon_ref=self.ref_dic['longitude']
        # 2. 经度连续性处理 (防止跨越 180° 线导致裁剪失败)
        # 如果经度跨度极大，说明跨越了日界线，统一转为 0-360 度处理
        if (np.nanmax(self.lon) - np.nanmin(self.lon) > 300) or \
           (np.nanmax(_lon_ref) - np.nanmin(_lon_ref) > 300):
            lon_tar_work = np.mod(self.lon, 360)
            lon_ref_work = np.mod(_lon_ref, 360)
            is_cross_180 = True
        else:
            lon_tar_work = self.lon
            lon_ref_work = _lon_ref
            is_cross_180 = False

        # 3. 计算共同区域 (Bounding Box)
        south_area = max(np.nanmin(self.lat), np.nanmin(_lat_ref))
        north_area = min(np.nanmax(self.lat), np.nanmax(_lat_ref))
        west_area = max(np.nanmin(lon_tar_work), np.nanmin(lon_ref_work))
        east_area = min(np.nanmax(lon_tar_work), np.nanmax(lon_ref_work))

        # 检查是否有重叠
        if south_area >= north_area or west_area >= east_area:
            logging.info(">>>>No overlapping areas at all<<<<")
            return 0

        # 4. 目标传感器裁剪
        loc1 = np.where((self.lat >= south_area) & (self.lat <= north_area) &
                        (lon_tar_work >= west_area) & (lon_tar_work <= east_area))
        
        if loc1[0].size < 100:
            logging.info(">>>>Overlapping area too small<<<<")
            return 0

        up, low = np.min(loc1[0]), np.max(loc1[0]) + 1
        left, right = np.min(loc1[1]), np.max(loc1[1]) + 1

        # 5. 角度数组自动填充
        for attr in ['sza', 'vza', 'saa', 'vaa']:
            val = getattr(self, attr)
            if isinstance(val, (float, int, np.number)):
                # 注意：这里用 self.lat 的原始形状进行填充
                setattr(self, attr, np.full(self.lat.shape, val))

        # 执行目标传感器裁剪
        self.sza = self.sza[up:low, left:right]
        self.vza = self.vza[up:low, left:right]
        self.saa = self.saa[up:low, left:right]
        self.vaa = self.vaa[up:low, left:right]
        self.lat = self.lat[up:low, left:right]
        self.lon = self.lon[up:low, left:right]
        self.Lt = self.Lt[up:low, left:right, :]

        # --- 新增：几何条件筛选 & 异常值处理 ---
        # 瑞利散射查找表通常要求 SZA < 75, VZA < 65，此处进行掩膜处理
        geom_mask = (self.sza < 75) & (self.vza < 65)
        
        # 将几何不达标或辐射亮度饱和 (>600) 的点设为 NaN
        for b in range(self.Lt.shape[2]):
            self.Lt[:, :, b][~geom_mask] = np.nan
            self.Lt[:, :, b][self.Lt[:, :, b] > 600] = np.nan

        # 6. 参考传感器裁剪
        loc2 = np.where((_lat_ref >= south_area) & (_lat_ref <= north_area) & 
                        (lon_ref_work >= west_area) & (lon_ref_work <= east_area))
        
        if loc2[0].size < 100:
            logging.info(">>>>Reference overlapping area too small<<<<")
            return 0
            
        up2, low2 = np.min(loc2[0]), np.max(loc2[0]) + 1
        left2, right2 = np.min(loc2[1]), np.max(loc2[1]) + 1

        for key in self.ref_dic:
            if isinstance(self.ref_dic[key], np.ndarray):
                self.ref_dic[key] = self.ref_dic[key][up2:low2, left2:right2]
        self.lat_ref=self.ref_dic['latitude']
        self.lon_ref=self.ref_dic['longitude']
        return 2

    def get_img_info(self, infile):
        """
        1.读取影像相关的信息
        Returns:
        """
        image_info = read_img_info.get(infile=infile, sensor_id=self.sensorID, block_size=self.block_size_rows)
        (self.data_Iterator, self.year, self.month, self.day, self.num_443, self.num_490, self.num_520, self.num_555,
         self.num_670, self.nirs_num, self.nirl_num, self.nwvis, self.red, self.rows_img, self.columns_img) = image_info

        date = datetime.datetime.strptime(str(self.year) + str(self.month) + str(self.day), "%Y%m%d")
        self.doy = int(date.strftime("%j"))
        self.fsol = esdist(self.doy)
        self.FoBAR = self.Fo * self.fsol
        self.Fo_ = self.FoBAR.reshape((1, 1, -1))
        print("correcting coefficient of solar-earth distance: " + str(self.fsol)[0:5])

    def get_refimg_info(self, infile):
        self.ref_dic  = read_aerosol_seadas.info(file=infile).run()
        self.rhoa_nirl = self.ref_dic['La_nirl'] * np.pi * self.fsol / (self.ref_dic['F0_nirl'] * np.cos(np.deg2rad(self.ref_dic['solz'])))
        self.ref_dic['rhoa_nirl'] = self.rhoa_nirl
    
    def meteor_para_and_gas_absorb(self):
        
        # 定义需要插值的变量列表
        interp_keys = ['pressure', 'ozone', 'no2_strat', 'no2_tropo', 'water_vapor']
        interp_results = interp(lat_ref=self.lat_ref, lon_ref=self.lon_ref, lat=self.lat, lon=self.lon, interp_keys=interp_keys, dataset=self.ref_dic, method='linear')
        
        # 解包插值结果
        self.pressure, ozone, no2_strat, tropo_no2, water_vapor = [interp_results[k] for k in interp_keys]

        # 3. 瑞利光学厚度的气压校正
        # /* Pressure correct the Rayleigh optical thickness */
        factor =self.pressure / 1013.25
        taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * self.Tau_r.reshape(1, 1, -1)
        # 4. 臭氧透过率校正，臭氧从欧洲下载，详情见函数内部。交叉定标直接从参考文件中读取，以保证一致性
        # print("computing gas absorbing transmittance: o3, no2, co2, h2o...")
        tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=self.sza, vza=self.vza,
                                                                       koz=self.k_oz, concentration=ozone)
        tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=self.k_no2, sza=self.sza, vza=self.vza,
                                                                          strat_no2=no2_strat, trop_no2=tropo_no2)
        tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=self.t_co2, sza=self.sza, vza=self.vza)
        tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=water_vapor, sza=self.sza,
                                                                          vza=self.vza, zia_table=self.Zia_table)
        tg_sol = tg_solar_o3 * tg_solar_no2 * tg_solar_co2 * tg_solar_h2o  # 其它吸收暂时不考虑
        tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_sensor_co2 * tg_sensor_h2o
        return taur, tg_sol, tg_sen

    def whitecap(self):
        # 5. 白帽反射
        # 这个白帽计算使用原始数据大小，能够有效提高计算速度，但是其中的taur是未经过气压校正的，影响应该不大

        # # 定义需要插值的变量列表
        # interp_results = interp(lat_ref=self.lat_ref, lon_ref=self.lon_ref, lat=self.lat, lon=self.lon, interp_keys=['windspeed'], dataset=self.ref_dic, method='linear')
        # # 解包插值结果
        # windspeed = interp_results['windspeed']
   
        
        rho_wc = whitecap_rad.calculate(U10=self.windspeed, bands=self.bands)
        t_sen = np.empty(shape=self.Lt.shape)
        t_sol = np.empty_like(t_sen)
        tLf = np.empty_like(t_sen)
        mu = np.cos(np.deg2rad(self.sza))
        mu0 = np.cos(np.deg2rad(self.vza))
        for i in range(self.bands.size):
            t_sen[:, :, i] = np.exp(-0.5 * self.Tau_r[i] / mu0)
            t_sol[:, :, i] = np.exp(-0.5 * self.Tau_r[i] / mu)
            tLf[:, :, i] = rho_wc[:, :, i] * t_sol[:, :, i] * t_sen[:, :, i] * self.FoBAR[i] * mu / np.pi
        return tLf, t_sen, t_sol

    def rayleigh(self):
        # 6. 移除瑞利贡献
        # print("computing rayleigh scattering radaince...")

        # 定义需要插值的变量列表
        interp_keys=['windspeed','windangle','pressure']
        interp_results = interp(lat_ref=self.lat_ref, lon_ref=self.lon_ref, lat=self.lat, lon=self.lon, interp_keys=interp_keys, dataset=self.ref_dic, method='linear')
        # 解包插值结果
        self.windspeed, self.windangle, self.pressure = [interp_results[key] for key in interp_keys]

        lr = ray_rad_idl.ray_rad(press=self.pressure, sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
                                 windspeed=self.windspeed, doy=self.doy, rayleigh_infos=self.rayleigh_lut_info,
                                 F0=self.Fo, bands=self.bands)
        scaleRayleigh = 1.0 - np.exp(-self.sensor_alt / 10)
        lr = lr * scaleRayleigh
        return lr

    def aerosol(self):
        # 7.气溶胶贡献
        # 通过插值保证相同的地理位置,将参考传感器的观测几何信息插入到目标传感器的位置

        # 定义需要插值的变量列表
        interp_keys=['rhoa_nirl','aer_model_ratio','aer_model_max','aer_model_min']
        interp_results = interp(lat_ref=self.lat_ref, lon_ref=self.lon_ref, lat=self.lat, lon=self.lon, interp_keys=interp_keys, dataset=self.ref_dic, method='nearest')
        # 解包插值结果
        rhoa_nirl, delta, aermod_up_idx, aermod_low_idx = [interp_results[key] for key in interp_keys]

        # 定义需要插值的变量列表
        interp_keys=['solz','senz','sola','sena']
        interp_results = interp(lat_ref=self.lat_ref, lon_ref=self.lon_ref, lat=self.lat, lon=self.lon, interp_keys=interp_keys, dataset=self.ref_dic, method='nearest')
        # 解包插值结果
        self.sza_ref, self.vza_ref, self.saa_ref, self.vaa_ref = [interp_results[key] for key in interp_keys]

        self.vza_ref[self.vza_ref > 80] = 80
        self.vza_ref[self.vza_ref < 0] = 0
        self.sza_ref[self.sza_ref > 80] = 80
        self.sza_ref[self.sza_ref < 0] = 0
        self.vza[self.vza > 80] = 80
        self.vza[self.vza < 0] = 0
        self.sza[self.sza > 80] = 80
        self.sza[self.sza < 0] = 0

        # 8.气溶胶反射
        aerosol = aerosol_rad.cross_calibration(delta=delta, rhoa_nirl=rhoa_nirl, sza=self.sza,
                                                vza=self.vza, saa=self.saa, vaa=self.vaa, aer_model_max=aermod_up_idx,
                                                aer_model_min=aermod_low_idx, aerosol_models_info=self.aerosol_lut_info,
                                                bands=self.bands, nirl_num=self.nirl_num, sza_ref=self.sza_ref,
                                                vza_ref=self.vza_ref, saa_ref=self.saa_ref, vaa_ref=self.vaa_ref,
                                                F0=self.Fo_, pressure=self.pressure, taur=self.Tau_r)
        if aerosol is None:
            print("没有有效的像元")
            return None, None, None, None
        La, t_sensor, t_solar, taua = aerosol["La"], aerosol["t_sensor"], aerosol["t_solar"], aerosol["taua"]

        La[:, :, self.nirs_num] = La[:, :, self.nirs_num]

        return La, t_sensor, t_solar, taua

    def glint(self):
        wd_rad = np.deg2rad(self.windangle)
        TLg = getglint.main_exec(sza=self.sza, vza=self.vza, vaa=self.vaa, saa=self.saa,
                                 taur=self.taur, La=self.La, F0=self.Fo_, windspeed=self.windspeed,
                                 winddirection=wd_rad, taua=self.taua, iter_num=1, mode=2)
        return TLg

    def water(self):
        # 8. 离水辐射
        # 定义需要插值的变量列表
        interp_keys=['Rrs']
        interp_results = interp(lat_ref=self.lat_ref, lon_ref=self.lon_ref, lat=self.lat, lon=self.lon, interp_keys=interp_keys, dataset=self.ref_dic, method='nearest')
        # 解包插值结果
        Rrs = [interp_results[key] for key in interp_keys][0]

        Rrs_tar = Rrs[:, :, self.sensorID_ref_bands]
        # Rrs_tar = np.zeros(shape=(self.sza.shape[0], self.sza.shape[1], Rrs.shape[2]))
        # for j in range(Rrs.shape[2]):
        #     rrs = Rrs[:, :, j]
        #     Rrs_tar[:, :, j] = interpolate.griddata((self.lat_ref.flatten(), self.lon_ref.flatten()), rrs.flatten(),
        #                                             (self.lat, self.lon), method='linear')

        nLw = Rrs_tar * self.Fo_
        mu0 = np.cos(np.deg2rad(self.sza)).reshape(self.rows_org, self.columns_org, 1)
        Lw = nLw * self.t_solar * self.tg_sol * mu0 / self.fsol
        tLw = Lw / self.tg_sol * self.t_sensor
        return tLw

    def geomask(self):
        geomask = mask_crossCalibration.mask_2(sza=self.sza, saa=self.saa, vza=self.vza, vaa=self.vaa,
                                               sza_ref=self.sza_ref, saa_ref=self.saa_ref, vza_ref=self.vza_ref,
                                               vaa_ref=self.vaa_ref)
        return geomask

    def output(self) -> str:
        outfile = self.outfile
        if os.path.exists(outfile):
            os.remove(outfile)
        print(">>>>>>>>>>>>生成的交叉定标文件:{}".format(os.path.basename(outfile)))
        f_new = h5py.File(outfile, 'a')
        f_new.attrs.create('target file', os.path.basename(self.output_info["tar_file"]), shape=(1,), dtype='S80')
        f_new.attrs.create('reference file', os.path.basename(self.output_info["ref_file"]), shape=(1,), dtype='S80')

        f_new.attrs.create('product time', datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), shape=(1,),
                           dtype='S80')
        f_new.attrs.create('author', 'Li Wenkai', shape=(1,), dtype='S30')
        f_new.attrs.create('email', 'lwk1542@scsio.ac.cn', shape=(1,), dtype='S26')
        f_new.attrs.create('method', 'Gordon; Wang; Hu;', shape=(1,), dtype='S26')
        f_new.attrs.create('fsol', self.fsol, shape=(1,), dtype='S26')

        (rows, columns) = self.tLf[:, :, 0].shape
        GeoData = f_new.create_group("Geophysical Data")
        for k, band in enumerate(self.bands):
            GeoData.create_dataset('tLf_' + str(band), (rows, columns), dtype='f', data=self.tLf[:, :, k])
            GeoData.create_dataset('TLg_' + str(band), (rows, columns), dtype='f', data=self.TLg[:, :, k])
            GeoData.create_dataset('La_' + str(band), (rows, columns), dtype='f', data=self.La[:, :, k])
            GeoData.create_dataset('Lr_' + str(band), (rows, columns), dtype='f', data=self.Lr[:, :, k])
            GeoData.create_dataset('tLw_' + str(band), (rows, columns), dtype='f', data=self.tLw[:, :, k])
            GeoData.create_dataset('t_sen_' + str(band), (rows, columns), dtype='f', data=self.t_sensor[:, :, k])
            GeoData.create_dataset('t_sol_' + str(band), (rows, columns), dtype='f', data=self.t_solar[:, :, k])
            GeoData.create_dataset('tg_sol_' + str(band), (rows, columns), dtype='f', data=self.tg_sol[:, :, k])
            GeoData.create_dataset('tg_sen_' + str(band), (rows, columns), dtype='f', data=self.tg_sen[:, :, k])
            GeoData.create_dataset('Lt_simu_' + str(band), (rows, columns), dtype='f', data=self.Lt_simu[:, :, k])
            GeoData.create_dataset('DN_' + str(band), (rows, columns), dtype='f', data=self.Lt[:, :, k])
        GeoData.create_dataset('sza_targetsensor', (rows, columns), dtype='f', data=self.sza)
        GeoData.create_dataset('vza_targetsensor', (rows, columns), dtype='f', data=self.vza)
        GeoData.create_dataset('saa_targetsensor', (rows, columns), dtype='f', data=self.saa)
        GeoData.create_dataset('vaa_targetsensor', (rows, columns), dtype='f', data=self.vaa)
        GeoData.create_dataset('sza_referencesensor', (rows, columns), dtype='f', data=self.sza_ref)
        GeoData.create_dataset('vza_referencesensor', (rows, columns), dtype='f', data=self.vza_ref)
        GeoData.create_dataset('saa_referencesensor', (rows, columns), dtype='f', data=self.saa_ref)
        GeoData.create_dataset('vaa_referencesensor', (rows, columns), dtype='f', data=self.vaa_ref)
        NavData = f_new.create_group("Navigation Data")
        para_name = ['lat', 'lon']
        for k, nav in enumerate([self.lat, self.lon]):
            NavData.create_dataset(para_name[k], (rows, columns), dtype='f', data=nav)

        mask_ = f_new.create_group("Mask")
        mask_.create_dataset("geo_mask", (rows, columns), dtype='f', data=self.mask)
        f_new.close()
        return outfile

def find_files_in_time_window(base_dir, date_str, time_str, window_minutes=180):
    """
    base_dir: 文件存放目录
    date_str: 格式为 '2021001' (YYYYDDD)
    time_str: 格式为 '0040' (HHMM)
    """
    ref_dir = Path(base_dir)
    
    # 1. 将输入的时间字符串转为 datetime 对象
    target_time = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
    
    # 2. 定义时间窗口边界
    start_window = target_time - datetime.timedelta(minutes=window_minutes)
    end_window = target_time + datetime.timedelta(minutes=window_minutes)
    
    search_doy = target_time.strftime("%Y%j")

    search_patterns = [f"MYD021KM.A{search_doy}.*_seadas_L2.hdf"]
    if start_window.day != target_time.day:
        search_patterns.append(f"MYD021KM.A{start_window.strftime('%Y%j')}.*_seadas_L2.hdf")
    if end_window.day != target_time.day:
        search_patterns.append(f"MYD021KM.A{end_window.strftime('%Y%j')}.*_seadas_L2.hdf")

    matched_files = []
    for pattern in set(search_patterns):
        for file_path in ref_dir.glob(pattern):
            try:
                # 文件名: MYD021KM.A2021046.2000.061...
                parts = file_path.name.split('.')
                f_doy_str = parts[1][1:] # 提取 2021046
                f_time_str = parts[2][:4] # 提取 2000，强制截断可能存在的秒数
                
                # 将文件名里的【年DOY时分】转回 datetime
                file_datetime = datetime.datetime.strptime(f"{f_doy_str}{f_time_str}", "%Y%j%H%M")
                
                # 5. 判断是否在窗口内
                if start_window <= file_datetime <= end_window:
                    diff = abs((file_datetime - target_time).total_seconds())
                    matched_files.append((file_path, diff))
                    
            except (IndexError, ValueError):
                continue

    # 6. 按时间差排序
    matched_files.sort(key=lambda x: x[1])
    
    return [f[0] for f in matched_files]



def cross():
    inpath='/work2/home_lwk/project/2025-1st-marinePICS/data/process/hy1d/'
    #inpath='/work2/home_lwk/project/2025-1st-marinePICS/data/process/test2/'
    infiles = sorted(list(Path(inpath).rglob("H1D_OPER_OCT_L1B_*_10.h5")),key=lambda x: x.name)
    reference_file_dir='/work2/home_lwk/project/2025-1st-marinePICS/data/process/modisa/'

    father_dir = "/work2/home_lwk/project/2025-1st-marinePICS/data/process/calibration/"

    for i, targetFile in enumerate(infiles): # H1D_OPER_OCT_L1B_20210102T013000_20210102T013500_02952_10.h5
        target_file_date = os.path.splitext(os.path.basename(targetFile))[0]
        # 使用下划线分割目标文件名中的日期部分
        # 格式: H1D_OPER_OCT_L1B_20210102T013000_20210102T013500_02952_10.h5
        # 提取日期: 20210102 -> 2021_01_02
        date_part = target_file_date.split('_')[4]
        year, month, day = date_part[:4], date_part[4:6], date_part[6:8]
        hour, minute = date_part[9:11], date_part[11:13]
        formatted_date = f"{year}{month}{day}"
        formatted_time=f"{hour}{minute}"

        # 使用新的函数查找匹配的文件
        # print(os.path.basename(targetFile))
        logging.info(f"--INFO--target:{os.path.basename(targetFile)}")
        reference_file_matches = find_files_in_time_window(reference_file_dir, formatted_date, formatted_time)

        for reference_file in reference_file_matches:
            logging.info(f"-----INFO-----refernece:{os.path.basename(reference_file)}")
             # 1. 优化文件名生成 

            tar_p = Path(targetFile)
            ref_p = Path(reference_file)
            id_parts = tar_p.name.split("_")[0]
            
            ref_stem = ref_p.stem
            outfile = Path(father_dir) / f"{id_parts}_{formatted_date}T{formatted_time}_crossCalibration.h5"
            
            if outfile.exists():
                logging.info(f"####INFO#### File already exists, skipping: {outfile.name}")
                continue

            simulate = SimulationLtoa(target_file=targetFile, outfile=outfile, ref_file=reference_file)
            toafile = simulate.run_main()

if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"corss_calibration_hy1dcocts_{timestamp}.log"
    log_path = script_dir / log_filename
    setup_logging(log_path)
    logging.info("程序启动，环境初始化完成")
    cross()
