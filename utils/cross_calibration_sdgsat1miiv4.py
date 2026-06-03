# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/13 10:49
@FileName: cross_calibration_sdgsat1miiv3.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""

import h5py
from scipy import interpolate
import cv2
import numpy as np
import datetime
import os
import glob
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AC_l2gen.utils import scene_time, esdist
from AC_l2gen.sensor import read_img_info
from AC_l2gen.sharepy import read_aerosol_seadas, read_tif
from AC_l2gen.l2gen import atmosphericParameter, gas_transmittance, aerosol_rad, mask_crossCalibration,\
    whitecap_rad, read_lut, getglint,  ray_rad_idl


class SimulationLtoa(object):
    def __init__(self,  target_file: str, ref_file: str, farther_dir: str):
        """
        Args:

        """
        # ++++++++++++++++++++++++++++++++++需要设置的参数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.target_file = target_file
        self.sensorID = "sdgsat1mii"
        # self.sensorID_ref = sensorid_ref  # s3b_olci,terra_modis,s3a_olci
        self.sensor_alt = 550
        self.sensorID_ref_bands = None
        self.fartherdir = farther_dir
        self.block_size_rows = None  # 后面设置一次全部读取
        self.ref_file = ref_file
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        self.fqfile = r"share" + os.sep + "common" + os.sep + 'morel_fq.h5'
        date_str = scene_time(self.target_file)
        self.pre_parameters(date_str)
        # 读取参考影像
        self.get_refimg_info(infile=self.ref_file)
        # 读取目标影像
        # 二. 获取影像相关的信息
        self.get_img_info(self.target_file)
        
        self.output_info = {"tar_file": self.target_file, "ref_file": self.ref_file}

        # for self.block_num, d_i_temp in enumerate(self.data_Iterator):  # 每个传感器的迭代器数据内容可能不一样，针对性处理
            # (data, gains, offsets, self.lon, self.lat, self.vaa, self.vza, self.saa, self.sza) = d_i_temp
            
            # "读取的依然是一整个文件"
            # self.Lt = data * gains.reshape(1, 1, -1) * 1. + offsets.reshape(1, 1, -1)

        # (self.rows_org, self.columns_org) = self.Lt[:, :, 0].shape

        print("***********参考影像{}".format(os.path.basename(self.ref_file)))
        # 1. 气象数据加载/计算气体吸收
        self.taur, self.tg_sol, self.tg_sen = self.meteor_para_and_gas_absorb()
        # 2. 白帽反射
        self.tLf, self.t_sen, self.t_sol = self.whitecap()
        # 3.瑞利
        self.Lr = self.rayleigh()
        # 4气溶胶
        self.La, self.t_sensor, self.t_solar, self.taua = self.aerosol()
        if self.La is None:
            return []
        # 5. 耀斑反射
        self.TLg = self.glint()
        # 6. 离水
        self.tLw = self.water()
        # LTOA
        airmass = 1 / np.cos(np.deg2rad(self.sza)) + 1 / np.cos(np.deg2rad(self.vza))
        # 7.氧气校正：
        t_o2 = gas_transmittance.oxygen_aer(airmass)
        temp = self.TLg + self.La + self.tLw
        temp[:, :, self.nirs_num] = temp[:, :, self.nirs_num] / t_o2
        self.Lt_simu = (self.tLf + self.Lr + temp) * self.tg_sol * self.tg_sen
        self.mask = self.geomask()

        outfile_name = self.output()

        return outfile_name

    def pre_parameters(self, date_str):
        """
        需要手动调整的参数
        """
        # self.south, self.north, self.west, self.east = self.limit[1], self.limit[3], self.limit[0], self.limit[2]
        self.outdir = self.fartherdir + os.sep + "result"
        if 'S3A' in self.ref_file:
            self.sensorID_ref = "s3a_olci"
            self.sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
        if 'S3B' in self.ref_file:
            self.sensorID_ref = "s3b_olci"
            self.sensorID_ref_bands = [0, 2, 3, 5, 7, 15, 16]
        self.ref_filedir = self.fartherdir + os.sep + 'reference'  # 数据路径

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

    def get_img_info(self, infile):
        """
        1.读取影像相关的信息
        Returns:
        """
        # image_info = read_img_info.get(infile=infile, sensor_id=self.sensorID, block_size=self.block_size_rows)
        # (self.sza, self.vza, self.saa, self.vaa, self.gains, self.bias, self.data_Iterator, self.year, self.month,
        #  self.day, self.num_443, self.num_490, self.num_520, self.num_555, self.num_670, self.nirs_num, self.nirl_num,
        #  self.nwvis, self.red) = image_info
        from sensor import sdgsat1mii
        
        dirname = os.path.dirname(infile)
        basename = os.path.basename(infile)
        name_id = "_".join(basename.split("_")[0:7])
        calib_file_path = glob.glob(dirname+os.sep+name_id + "*.calib.xml")[0]
        meta_file_path = glob.glob(dirname+os.sep+name_id + "*.meta.xml")[0]
        # calib_file_path=infile.replace(".tif", ".calib.xml")id_ = subdir.split("_")
        #         id = "_".join(id_[0:-1])
        # meta_file_path = infile.replace(".tif", ".meta.xml")
        gains, bias = sdgsat1mii.calib_xml(calib_file_path)
        time, saa_, sza_, roll, pitch, yaw = sdgsat1mii.meta_xml(meta_file_path)
        year, month, day, hour, minute, second = \
            int(time[0:4]), int(time[5:7]), int(time[8:10]), int(time[11:13]), int(time[14:16]), int(float(time[17:]))

        self.sza = np.full_like(self.aermod_low_idx, fill_value=sza_)
        self.saa = np.full_like(self.sza, fill_value=saa_)
        self.vza = np.full_like(self.sza, fill_value=5)
        self.vaa = np.full_like(self.sza, fill_value=-100)

        
        # image_info = read_img_info.get(infile=infile, sensor_id=self.sensorID, block_size=self.block_size_rows)
        # (self.data_Iterator, self.year, self.month, self.day, self.num_443, self.num_490, self.num_520, self.num_555,
        #  self.num_670, self.nirs_num, self.nirl_num, self.nwvis, self.red, self.rows_img, self.columns_img) = image_info

        date = datetime.datetime.strptime(str(self.year) + str(self.month) + str(self.day), "%Y%m%d")
        self.doy = int(date.strftime("%j"))
        self.fsol = esdist(self.doy)
        self.FoBAR = self.Fo * self.fsol
        self.Fo_ = self.FoBAR.reshape((1, 1, -1))
        print("correcting coefficient of solar-earth distance: " + str(self.fsol)[0:5])

    def get_refimg_info(self, infile):
        # [self.Rrs, self.delta, La_nirl, Fo_nirl, self.lat_ref, self.lon_ref, self.sza_ref, self.vza_ref,
        #  self.saa_ref, self.vaa_ref, self.aermod_up_idx, self.aermod_low_idx] = read_aerosol_seadas.info(
        #     file=infile).run()
        self.ref_dic  = read_aerosol_seadas.info(file=infile).run()
        self.lat_ref=self.ref_dic['latitude']
        self.lon_ref=self.ref_dic['longitude']
        self.sza_ref=self.ref_dic['solz']
        self.vza_ref=self.ref_dic['senz']
        self.saa_ref=self.ref_dic['sola']
        self.vaa_ref=self.ref_dic['sena']
        self.aermod_up_idx=self.ref_dic['aer_model_max']
        self.aermod_low_idx=self.ref_dic['aer_model_min']
        self.delta=self.ref_dic['aer_model_ratio']
        
        self.rhoa_nirl = self.ref_dic['La_nirl'] * np.pi * self.fsol / (self.ref_dic['F0_nirl'] * np.cos(np.deg2rad(self.sza_ref)))

    def meteor_para_and_gas_absorb(self):
        # print("load atmospheric parameters: pressure, O3, NO2, water vapor, reality humidity, wind etc:...")
        # self.winds_peed, self.winddirection, self.pressure, o3, self.rh, water_vapor, strat_no2, trop_no2, self.taua = \
        #     atmosphericParameter.get(Lon=self.lon, Lat=self.lat, year=self.year, month=self.month,
        #                              day=self.day, time='03:00')
        self.winds_peed=self.ref_dic['windspeed']
        self.winddirection=self.ref_dic['windangle']
        self.pressure=self.ref_dic['pressure']

        # 3. 瑞利光学厚度的气压校正
        # /* Pressure correct the Rayleigh optical thickness */
        factor = self.pressure / 1013.25
        taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * self.Tau_r.reshape(1, 1, -1)
        # 4. 臭氧透过率校正，臭氧从欧洲下载，详情见函数内部。交叉定标直接从参考文件中读取，以保证一致性
        # print("computing gas absorbing transmittance: o3, no2, co2, h2o...")
        tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=self.sza, vza=self.vza,
                                                                       koz=self.k_oz, concentration=self.ref_dic['ozone'])
        tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=self.k_no2, sza=self.sza, vza=self.vza,
                                                                          strat_no2=self.ref_dic['no2_strat'], trop_no2=self.ref_dic['tropo_no2'])
        tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=self.t_co2, sza=self.sza, vza=self.vza)
        tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=self.ref_dic['water_vapor'], sza=self.sza,
                                                                          vza=self.vza, zia_table=self.Zia_table)
        tg_sol = tg_solar_o3 * tg_solar_no2 * tg_solar_co2 * tg_solar_h2o  # 其它吸收暂时不考虑
        tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_sensor_co2 * tg_sensor_h2o
        return taur, tg_sol, tg_sen

    def whitecap(self):
        # 5. 白帽反射
        # 这个白帽计算使用原始数据大小，能够有效提高计算速度，但是其中的taur是未经过气压校正的，影响应该不大
        rho_wc = whitecap_rad.calculate(U10=self.winds_peed, bands=self.bands)
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

        lr = ray_rad_idl.ray_rad(press=self.pressure, sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
                                 windspeed=self.winds_peed, doy=self.doy, rayleigh_infos=self.rayleigh_lut_info,
                                 F0=self.Fo, bands=self.bands)

        scaleRayleigh = 1.0 - np.exp(-self.sensor_alt / 10)
        lr = lr * scaleRayleigh
        # lr = np.empty_like(self.Lt)
        # for band_num in range(self.bands.size):
        #     lr[:, :, band_num] = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
        #                                               lr[:, :, band_num].flatten(), (self.lat, self.lon),
        #                                               method='nearest')
        return lr

    def aerosol(self):
        # 7.气溶胶贡献
        # 通过插值保证相同的地理位置,将参考传感器的观测几何信息插入到目标传感器的位置
        # Rrs_tar[:, :, j] = interpolate.griddata((self.lat_ref.flatten(), self.lon_ref.flatten()), rrs.flatten(),
        #                                             (self.lat, self.lon), method='linear')

        # 8.气溶胶反射
        aerosol = aerosol_rad.cross_calibration(delta=self.delta, rhoa_nirl=self.rhoa_nirl, sza=self.sza,
                                                vza=self.vza, saa=self.saa, vaa=self.vaa, aer_model_max=self.aermod_up_idx,
                                                aer_model_min=self.aermod_low_idx, aerosol_models_info=self.aerosol_lut_info,
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
        wd_rad = np.deg2rad(self.winddirection)
        TLg = getglint.main_exec(sza=self.sza, vza=self.vza, vaa=self.vaa, saa=self.saa,
                                 taur=self.taur, La=self.La, F0=self.Fo_, windspeed=self.winds_peed,
                                 winddirection=wd_rad, taua=self.taua, iter_num=1, mode=2)
        return TLg

    def water(self):
        # 8. 离水辐射
        Rrs = self.Rrs_ref[:, :, self.sensorID_ref_bands]
        Rrs_tar = np.zeros(shape=(self.sza.shape[0], self.sza.shape[1], Rrs.shape[2]))
        nLw = Rrs * self.Fo_
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

# if __name__ == '__main__':
#     SimulationLtoa().run_main()
