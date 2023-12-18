# -*- coding: utf-8 -*-
"""
@Time    : 2023/11/29 14:52
@FileName: main_ac.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np
import datetime
import os
import gc
import cv2
from utils import esdist, resize
from sensor import read_img_info
from utils import outfile_setting as output
from l2gen import atmosphericParameter, gas_transmittance, get_rhown_nir, whitecap_rad, get_chl, read_lut, getglint, brdf, aerosol_radV2, rayleigh_rad, \
    predefine


class Calcu(object):
    def __init__(self, filespath: str, sensorid: str = "hy1ccocts" or "fy3dmersi" or "sdgsat1mii"):
        # ++++++++++++++++++++++++++++++++++需要设置的参数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # self.sensorID = "hy1ccocts"  # "sdgsat1mii", "olcis3a"
        # self.filespath = r"C:\Users\DELL\Desktop\test\fy3dmersi"  # 数据路径
        self.sensorID = sensorid  # "sdgsat1mii", "olcis3a", "hy1ccocts", "fy3dmersi"
        self.filespath = filespath  # 数据路径
        self.package_match()
        self.sensor_alt = 720
        self.nonzeroNIR = "want_nirLw"
        self.resize = None  # 重采样的尺度
        self.block_size_rows = 3000  # 一次读取原始影像的行数;None全部读取
        self.rrc_out = True  # 是否输出相关结果
        self.rrs_out = True
        self.ltoa_out = True
        self.tLf_out = True
        self.TLg_out = True
        self.La_out = True
        self.tLf = True
        self.tg_out = True
        self.lr_out = True
        self.chl_out = False
        self.brdf_factor = None
        self.south, self.north, self.west, self.east = 21.9, 22.8, 113, 114.6  # 尚未使用
        self.nir_iter = True
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.columns_img = None  # 整景影像的列数
        self.rows_img = None  # 整景影像的行数
        self.columns_chunk = None  # 分块读取的数据列数
        self.rows_chunk = None
        self.vaa_chunk = None
        self.vza_chunk = None
        self.saa_chunk = None
        self.sza_chunk = None
        self.lon_chunk = None
        self.lat_chunk = None
        self.Lt_chunk = None
        self.lat = None
        self.lon = None
        self.aerosol_lut_info = None
        self.rayleigh_lut_info = None
        self.sensor_info = None
        self.infile = None
        self.fsol = None
        self.Tau_r = None
        self.rh = None
        self.winddirection = None
        self.wind_speed = None
        self.pressure = None
        self.taua = None
        self.rows_ext = None
        self.strat_no2 = None
        self.trop_no2 = None
        self.o3 = None
        self.water_vapor = None
        self.meteo_da = None
        self.Fo_ = None
        self.fqfile = None
        self.FoBAR = None
        self.chl = None
        self.sza = None
        self.vza = None
        self.saa = None
        self.vaa = None
        self.Lt = None
        self.Ltemp = None
        self.t_sol = None
        self.t_sen = None
        self.tg_sen = None
        self.tg_sol = None
        self.nLw = None
        self.taur = None
        self.block_num = None
        self.geo_group = None
        self.navi_group = None
        self.ds = None

    def run_main(self):
        """
        主函数：
        Returns:
        """
        # 一. 获取传感器相关的信息
        self.sensor_info, self.rayleigh_lut_info, self.aerosol_lut_info = read_lut.CommonVariable(
            sensor_id=self.sensorID).get()
        (self.bands, self.Fo, self.Tau_r, self.k_oz, self.t_co2, self.k_no2, self.Zia_table, self.awhite, self.aw,
         self.bbw, self.oobwv, self.ooblw, self.wed, self.waph) = self.sensor_info
        self.fqfile = r"../share" + os.sep + "common" + os.sep + 'morel_fq.h5'

        files = self.file_match()
        print("total {} images:".format(files.__len__()), files)
        for self.infile in files:
            print("starting processing {}".format(os.path.basename(self.infile)))
            starttime = datetime.datetime.now()
            # 二. 获取影像相关的信息
            data_Iterator = self.get_img_info(self.infile)
            if data_Iterator == 0: continue
            # 三. 建立需要输出的文件数据驱动
            self.out_varable()
            # 迭代器分块处理数据，节约内存开支
            self.rows_ext = 0  # 从第0行开始
            for self.block_num, d_i_temp in enumerate(data_Iterator):  # 每个传感器的迭代器数据内容可能不一样，针对性处理
                (data, gains, offsets, self.lon, self.lat, self.vaa, self.vza, self.saa, self.sza) = d_i_temp
                Lt = (data * gains.reshape((1, 1, -1)) * 1. + offsets.reshape((1, 1, -1)))
                self.Lt = self.unit(Lt)  # w/m-2
                (self.rows_chunk, self.columns_chunk) = Lt[:, :, 0].shape
                # self.Lt = self.cloud_land_mask(lt=Lt)
                del Lt, data, gains, offsets
                gc.collect()
                print("==========================processing {}~{} lines=================================".format(
                    self.block_size_rows * self.block_num, self.block_size_rows * self.block_num + self.rows_chunk))

                navi_para = ['longitude', "latitude", 'vza', 'vaa', 'sza', 'saa']
                for i, _temp in enumerate([self.lon, self.lat, self.vza, self.vaa, self.sza, self.saa]):
                    output.write(ds_group=self.navi_group, data_name=navi_para[i], chunk=_temp, rows_ext=self.rows_ext,
                                 columns=self.columns_img)
                del _temp, navi_para

                if self.ltoa_out:
                    for i, band_temp in enumerate(self.bands):
                        output.write(ds_group=self.geo_group, data_name="Lt_" + str(int(band_temp)),
                                     chunk=self.Lt[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                    del band_temp

                if self.resize:
                    # 是否重采样
                    re_temp = [re_ * 1.0 for re_ in [self.lat, self.lon, self.sza, self.saa, self.vza, self.vaa]]
                    [self.lat_chunk, self.lon_chunk, self.sza_chunk, self.saa_chunk, self.vza_chunk,
                     self.vaa_chunk] = re_temp
                    del re_temp
                    _ = resize.down_sample(lat=self.lat, lon=self.lon, sza=self.sza, saa=self.saa, vza=self.vza,
                                           vaa=self.vaa, resize=self.resize)
                    self.lat, self.lon, self.sza, self.saa, self.vza, self.vaa = _
                    del _
                else:
                    [self.lat_chunk, self.lon_chunk, self.sza_chunk, self.saa_chunk, self.vza_chunk,
                     self.vaa_chunk] = [self.lat, self.lon, self.sza, self.saa, self.vza, self.vaa]

                # 四. 气象数据加载/计算气体吸收
                self.meteor_parameter()
                taur, self.tg_sol, tg_sen = self.gas_absorb()
                if self.tg_out:
                    for i, band_temp in enumerate(self.bands):
                        output.write(ds_group=self.geo_group, data_name="tg_sol_" + str(int(band_temp)),
                                     chunk=self.tg_sol[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                        output.write(ds_group=self.geo_group, data_name="tg_sen_" + str(int(band_temp)),
                                     chunk=tg_sen[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                self.Ltemp = self.Lt / tg_sen / self.tg_sol
                # 卷云和极化校正：没做
                # 五. 白帽反射 # 基于瑞利光学厚度的上下和下行透过率
                tLf, t_sen, t_sol = self.whitecap()
                if self.tLf_out:
                    for i, band_temp in enumerate(self.bands):
                        output.write(ds_group=self.geo_group, data_name="tLf_" + str(int(band_temp)),
                                     chunk=tLf[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                self.Ltemp -= tLf
                del tLf
                # 六.瑞利
                lr = self.rayleigh()
                self.Ltemp -= lr
                if self.lr_out:
                    for i, band_temp in enumerate(self.bands):
                        output.write(ds_group=self.geo_group, data_name="Lr_" + str(int(band_temp)),
                                     chunk=lr[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                # del lr

                a_o2, t_o2 = self.oxygen_correct()
                self.Ltemp[:, :, self.nirs_num] *= t_o2
                # Rrc = (self.Lt-lr) / self.Fo_ / np.cos(np.deg2rad(
                #     self.sza_chunk.reshape((self.rows_chunk, self.columns_chunk, 1)))) / t_sen / t_sen
                # 云掩膜
                Rrc = np.pi * self.Ltemp / self.Fo_ / np.cos(
                    np.deg2rad(self.sza_chunk.reshape((self.rows_chunk, self.columns_chunk, 1)))) / t_sen / t_sen
                mask = self.cloud_land_mask(rhos=Rrc)
                self.Ltemp = self.Ltemp * mask
                Rrc=Rrc*mask
                if self.rrc_out:
                    # Rrc = np.pi * self.Ltemp / self.Fo_ / np.cos(
                    #     np.deg2rad(self.sza.reshape(self.rows_org, self.columns_org, 1)))
                    for i, band_temp in enumerate(self.bands):
                        output.write(ds_group=self.geo_group, data_name="rhos_" + str(int(band_temp)),
                                     chunk=Rrc[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                    del Rrc, band_temp
                    gc.collect()

                # 七.耀斑和近红外迭代气溶胶
                if self.rrs_out:
                    if self.nir_iter:
                        out_ = self.aerosol_iter_nir()
                        if out_.__len__() == 5:
                            [La, TLg, t_sen, t_sol, self.chl] = out_
                        else:
                            TLg, La = out_
                            print("No aerosol")
                    else:
                        TLg, La = np.zeros_like(self.Ltemp), np.zeros_like(self.Ltemp)
                    Lw = (self.Ltemp - TLg - La) / t_sen * self.tg_sol
                    mu0 = np.cos(np.deg2rad(self.sza_chunk)).reshape(self.rows_chunk, self.columns_chunk, 1)
                    nLw = Lw / t_sol / self.tg_sol / mu0 / self.fsol
                    brdf_factor = self.fq_brdf_correct(nLw=nLw)
                    nLw = nLw * brdf_factor
                    #  /* Compute final Rrs */
                    Rrs = nLw / self.Fo_
                    # /* Compute final chl from final nLw (needed for flagging) */
                    chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
                                                  b520=self.num_520, b555=self.num_555, b670=self.num_670,
                                                  sensorid=self.sensorID)
                    for i, band_temp in enumerate(self.bands):
                        output.write(ds_group=self.geo_group, data_name="Rrs_" + str(int(band_temp)),
                                     chunk=Rrs[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                    if self.La_out:
                        for i, band_temp in enumerate(self.bands):
                            output.write(ds_group=self.geo_group, data_name="La_" + str(int(band_temp)),
                                         chunk=La[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                            output.write(ds_group=self.geo_group, data_name="TLg_" + str(int(band_temp)),
                                         chunk=TLg[:, :, i], rows_ext=self.rows_ext, columns=self.columns_img)
                    chl[chl < 0] = np.nan
                    output.write(ds_group=self.geo_group, data_name="chlor_a",
                                 chunk=chl, rows_ext=self.rows_ext, columns=self.columns_img)
                self.rows_ext += self.rows_chunk
            endtime = datetime.datetime.now()
            print("====Total time to process this imagery: {} minutes======".format(
                round((endtime - starttime).seconds / 60), 1))
            print("====time : {}======".format(endtime))
            output.close(self.ds)

    def package_match(self):
        """添加一个函数，简化导入程序"""
        match self.sensorID:
            case "hy1ccocts":
                from sensor.hy1ccocts import mask, file
            case "hy1dcocts":
                from sensor.hy1dcocts import mask, file
            case "fy3dmersi":
                from sensor.fy3dmersi import mask, file
            case _:
                from sensor.hy1ccocts import mask, file
        return mask, file

    def file_match(self):
        mask, file = self.package_match()
        files = file.search_(path=self.filespath)
        return files

    def cloud_land_mask(self, rhos):
        mask, file = self.package_match()
        mask_matrix = mask.cloud_land_mask(rhos=rhos*1.)
        # mask_matrix = 1.
        return mask_matrix

    def get_img_info(self, infile):
        """
        1.读取影像相关的信息
        """
        image_info = read_img_info.get(infile=infile, sensor_id=self.sensorID, block_size=self.block_size_rows)
        if image_info.__len__() == 1:
            return 0
        (data_Iterator, self.year, self.month, self.day, self.num_443, self.num_490, self.num_520, self.num_555,
         self.num_670, self.nirs_num, self.nirl_num, self.nwvis, self.red, self.rows_img, self.columns_img) = image_info

        date = datetime.datetime.strptime(str(self.year) + str(self.month) + str(self.day), "%Y%m%d")
        doy = int(date.strftime("%j"))
        self.fsol = esdist(doy)
        self.FoBAR = self.Fo * self.fsol
        self.Fo_ = self.FoBAR.reshape((1, 1, -1))
        print("correcting coefficient of solar-earth distance: " + str(self.fsol)[0:5])
        return data_Iterator

    def whitecap(self):
        # 5. 移除白帽反射
        # 这个白帽计算使用原始数据大小，能够有效提高计算速度，但是其中的taur是未经过气压校正的，影响应该不大
        rho_wc = whitecap_rad.calculate(U10=self.wind_speed, bands=self.bands)
        self.t_sen = np.empty(shape=(rho_wc.shape[0], rho_wc.shape[1], self.bands.size))
        self.t_sol = np.empty_like(self.t_sen)
        self.tLf = np.empty_like(self.t_sen)
        mu = np.cos(np.deg2rad(self.sza))
        mu0 = np.cos(np.deg2rad(self.vza))
        for i in range(self.bands.size):
            self.t_sen[:, :, i] = np.exp(-0.5 * self.Tau_r[i] / mu0)
            self.t_sol[:, :, i] = np.exp(-0.5 * self.Tau_r[i] / mu)
            self.tLf[:, :, i] = rho_wc[:, :, i] * self.t_sol[:, :, i] * self.t_sen[:, :, i] * self.FoBAR[i] * mu / np.pi
        if self.resize:
            t_sen = resize.up_sample(data=self.t_sen, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                     lon_tar=self.lon_chunk)
            t_sol = resize.up_sample(data=self.t_sol, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                     lon_tar=self.lon_chunk)
            tLf = resize.up_sample(data=self.tLf, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                   lon_tar=self.lon_chunk)
            return tLf, t_sen, t_sol
        return self.tLf, self.t_sen, self.t_sol

    def rayleigh(self):
        # 6. 移除瑞利贡献
        # print("computing rayleigh scattering radaince...")
        # lr = rayleigh_rad_V201.rayleigh(raylut_info=self.rayleigh_lut_info, sza=self.sza,
        #                                 vza=self.vza, saa=self.saa, vaa=self.vaa,
        #                                 F0=self.FoBAR, windspeed=self.wind_speed,
        #                                 pressure=self.pressure)
        lr = rayleigh_rad.rayleigh(rayleigh_lut_path=self.rayleigh_lut_info, sza=self.sza, vza=self.vza, saa=self.saa,
                                   vaa=self.vaa, F0=self.FoBAR, windspeed=self.wind_speed, pressure=self.pressure)

        scaleRayleigh = 1.0 - np.exp(-self.sensor_alt / 10)
        self.lr = lr * scaleRayleigh
        if self.resize:
            lr = resize.up_sample(data=lr, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk, lon_tar=self.lon_chunk)
            return lr
        return self.lr

    def aerosol_iter_nir(self):
        # /* ------------------------------------------------------------------------------------------------------ */
        # /* Begin interations for aerosol with corrections for non-zero nLw(NIR)近红外波段不等于0 */
        # /* ------------------------------------------------------------------------------------------------------ */
        # / *Initialize tLw as surface + aerosol radiance * /
        # taua = 0.1
        # self.chl = np.full(shape=(self.rows_chunk, self.columns_chunk), fill_value=predefine.thresholds().seed_chl)
        ltemp = self.Ltemp * 1.
        if self.resize:
            ltemp = resize.down_sample_aerosol(lt=ltemp, resize=self.resize,
                                               band_nirs=self.nirs_num, band_nirl=self.nirl_num)
        chl = np.full(shape=ltemp[:, :, -1].shape, fill_value=predefine.thresholds().seed_chl)
        last_tLw_nir = np.zeros_like(ltemp)
        tLw_nir = np.zeros_like(ltemp)
        Rrs = np.zeros_like(ltemp)
        Rrs[:, :, self.num_555] = predefine.thresholds().seed_green  # 0假设
        Rrs[:, :, self.num_670] = predefine.thresholds().seed_red
        rows_, columns_ = self.sza.shape[0], self.sza.shape[1]
        mu0 = np.cos(np.deg2rad(self.sza)).reshape(rows_, columns_, 1)
        brdf_factor = np.ones_like(ltemp)

        #  /* Initialize iteration loop */
        last_iter = np.zeros_like(mu0[:, :, 0])  # 用于标识是否为最后一次迭代，1表示是，0表示否（继续迭代）

        iterx = np.zeros_like(last_iter)  # 每个像元的跌打次数
        last_refl_nir = np.full_like(last_iter, fill_value=100.)
        iter_reset = np.zeros_like(last_iter)
        iter_max = np.full_like(last_iter, fill_value=predefine.thresholds().aer_iter_max)
        loc_iter = np.full_like(mu0, fill_value=1.)  # 这个矩阵要乘以3维矩阵的，声明为3维
        tLw_final = np.full_like(ltemp, fill_value=np.nan)
        TLg_final = np.full_like(ltemp, fill_value=np.nan)
        La_final = np.full_like(ltemp, fill_value=np.nan)   # 最终计算的气溶胶结果
        t_sen_final = np.full_like(ltemp, fill_value=self.t_sen * 1.)
        t_sol_final = np.full_like(ltemp, fill_value=self.t_sol * 1.)
        La = np.full_like(ltemp, fill_value=np.nan)  # 本次计算的气溶胶结果
        TLg = np.full_like(ltemp, fill_value=np.nan)
        t_sen = np.full_like(ltemp, fill_value=np.nan)
        t_sol = np.full_like(ltemp, fill_value=np.nan)
        La_pre = np.full_like(ltemp, fill_value=np.nan)  # 上一次迭代计算的气溶胶结果
        TLg_pre = np.full_like(ltemp, fill_value=np.nan)
        t_sen_pre = np.full_like(ltemp, fill_value=np.nan)
        t_sol_pre = np.full_like(ltemp, fill_value=np.nan)
        cslp = 1. / (predefine.thresholds().ctop - predefine.thresholds().cbot)
        cint = -cslp * predefine.thresholds().cbot

        # while iter_num <= predefine.thresholds().glint_iter_max:
        wd_rad = np.deg2rad(self.winddirection)
        iter_num = 0
        while np.nanmin(last_iter) == 0:  # last_iter是一个数组，初始值全为0，用于记录每个像元自身是否达到停止迭代条件任何一个像元没有达到停止迭代条件，均需要继续迭代
            iterx = iterx + 1  # 迭代次数，是一个矩阵，记录每个像元的迭代次数（处于迭代过程中的像元）
            print(">>>>>>迭代计算气溶胶辐亮度，第{0}次".format(iter_num))
            # 迭代过程中的iter_num也用于记录迭代次数，不过它是一个数，记录的是这个迭代程序运行的次数
            # 耀斑福亮度估算, First, the measured Lt(λ) and the wind
            # M. Wang and S. Bailey,Correction of sun glint contamination on the SeaWiFS ocean and atmosphere products
            # 第一次其实只需要做出近红外波段的耀斑，助后面选出气溶胶模型
            # /* Initialize tLw as surface + aerosol radiance */
            # loc_iter表示是否参与计算，初始化全部为1，通过迭代后，不需要参与下一次计算的像元在 loc_iter中标记为np.nan
            tLw = ltemp * loc_iter   # 初始化ltemp，作为水体、气溶胶、耀斑的总信号，后面将逐项减去
            if np.nanmin(iterx) > 1:
                La_pre = La * 1.
                TLg_pre = TLg * 1.
                t_sol_pre = t_sol * 1.
                t_sen_pre = t_sen * 1.
            iter_num_glint = 2  # 用于计算耀斑的最大迭代次数，wang指出2次就够
            mode = 2  # mode表示计算耀斑时的气溶胶光学厚度数据来源，2表示从外部获取，1和0分别表示设置0.1的固定值或者依据wang的估算方法
            TLg = getglint.main_exec(iter_num=iter_num_glint, sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
                                     taur=self.taur, La=tLw*1., F0=self.FoBAR, windspeed=self.wind_speed,
                                     winddirection=wd_rad, taua=self.taua, mode=mode)
            tLw = ltemp - TLg
            # step 1:  对于近红外波段离水非0信号，通过IOP模型推导NIR离水信号
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>近红外迭代的关键过程>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # /* Adjust for non-zero NIR water-leaving radiances using IOP model */
            # 使用IOP模型、纯水吸收散射、rrs去估算近红外波段的离水辐亮度，这样就可以获得近红外波段的气溶胶福亮度
            # 第一次假设近红外波段的离水辐亮度为0，NIR波段全部为气溶胶信号。因此初始化设置RRS为0，可获得这一假设结果，并且实现程序
            rhown_nir = get_rhown_nir.get_rhown_eval(num_443=self.num_443, num_555=self.num_555,
                                                     num_670=self.num_670, nirs_num=self.nirs_num,
                                                     nirl_num=self.nirl_num, Rrs=Rrs, chl=chl, aw=self.aw,
                                                     bbw=self.bbw, bands=self.bands, fqfile=self.fqfile,
                                                     sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa)
            for ib in range(self.nirs_num, self.nirl_num + 1):
                # 更新近红外波段的离水信号
                tLw_nir[:, :, ib] = (rhown_nir[:, :, ib] / np.pi * self.Fo[ib] * mu0[:, :, 0] *
                                     t_sol[:, :, ib] * t_sen[:, :, ib]) / brdf_factor[:, :, ib]
                #  /* Iteration damping */
                tLw_nir[:, :, ib] = (1.0 - predefine.thresholds().df) * tLw_nir[:, :, ib] + (predefine.thresholds().df
                                                                                             * last_tLw_nir[:, :, ib])
                # /* Ramp-up ?*/
                tLw_nir[:, :, ib][(0 < chl) & (chl < predefine.thresholds().cbot)] = 0.
                loc_temp = (predefine.thresholds().cbot < chl) & (chl < predefine.thresholds().ctop)
                tLw_nir[:, :, ib][loc_temp] = tLw_nir[:, :, ib][loc_temp] * (cslp * chl + cint)[loc_temp]
                # /* Remove estimated NIR water-leaving radiance */
                tLw[:, :, ib] = tLw[:, :, ib] - tLw_nir[:, :, ib]
                tLw[:, :, ib][tLw[:, :, ib] < 0] = 0  # 避免过校正
                del loc_temp

            # step 2:到这里假设，通过离水的近红外信号和瑞利校正后的近红外信号，计算气溶胶的近红外信号
            l_nir1 = tLw[:, :, self.nirs_num] #- tLw_nir[:, :, self.nirs_num]
            l_nir2 = tLw[:, :, self.nirl_num] #- tLw_nir[:, :, self.nirl_num]
            # aero_out = aerosol_rad.atmos_corr(l_a_nir1=l_nir1, l_a_nir2=l_nir2, lon=self.lon, lat=self.lat,
            #                                   F0=self.FoBAR, bands=self.bands, taur=self.Tau_r,
            #                                   aerosol_models_info=self.aerosol_lut_info, pressure=self.pressure,
            #                                   sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
            #                                   winds_peed=self.wind_speed, relative_humidity=self.rh,
            #                                   nirl_num=self.nirl_num, nirs_num=self.nirs_num)
            aero_out = aerosol_radV2.calculate(l_a_nir1=l_nir1, l_a_nir2=l_nir2, lon=self.lon, lat=self.lat,
                                               F0=self.FoBAR, bands=self.bands, taur=self.Tau_r,
                                               aerosol_lut_filepath=self.aerosol_lut_info, pressure=self.pressure,
                                               sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
                                               winds_peed=self.wind_speed, relative_humidity=self.rh,
                                               nirl_num=self.nirl_num, nirs_num=self.nirs_num)
            if aero_out is None:
                if self.resize:
                    TLg_final = resize.up_sample(data=TLg, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                                 lon_tar=self.lon_chunk)
                return [La_final, TLg_final, t_sen_final, t_sol_final, self.chl]

            La, t_sen, t_sol, self.taua, aer1, aer2 = aero_out
            if iter_num == 0:
                La_pre = La * 1.
                TLg_pre = TLg * 1.
                t_sol_pre = t_sol * 1.
                t_sen_pre = t_sen * 1.
                tLw_pre = tLw * 1.
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<近红外迭代的关键过程<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            tLw = tLw - La
            Lw = tLw / t_sen * self.tg_sol
            nLw = Lw / t_sol / self.tg_sol / mu0 / self.fsol
            brdf_factor = self.fq_brdf_correct(nLw=nLw*1.)
            nLw = nLw * brdf_factor

            # step 3:气溶胶校正完成，计算新的Rrs和叶绿素，并进行一系列判断，是否为合理的Rrs，不合理的像元进行下一次迭代，后面根据新的Rrs估算近红外离水信息
            # /* Compute new estimated chlorophyll */
            refl_nir = Rrs[:, :, self.red]  ##？？？？？
            for ib in range(self.nirs_num, self.nirl_num + 1):
                last_tLw_nir[:, :, ib] = tLw_nir[:, :, ib]
            for ib in range(self.nwvis):
                Rrs[:, :, ib] = nLw[:, :, ib] / self.FoBAR[ib]
            chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
                                          b520=self.num_520, b555=self.num_555, b670=self.num_670,
                                          sensorid=self.sensorID)

            #    if we passed atmospheric correction but the spectral distribution of
            # Rrs is bogus (self.chl failed), assume this is a turbid-water case and
            # reseed iteration as if all 670 reflectance is from water.

            # step3.1 如果新计算的的叶绿素异常，那么这个位置的红光波段的反射率则不减去计算的La
            # if we already tried a reset, and still no convergence, force one last
            # pass with an assumption that all red radiance is water component, and
            # force iteration to end.  this will be flagged as atmospheric correction
            # failure, but a qualitatively useful retrieval may still result.

            abnormal_chl_id_1 = ((chl == predefine.thresholds().chlbad) & (iter_reset == 1) & (iterx < iter_max))  # 原判断
            chl[abnormal_chl_id_1] = 10
            iter_reset[abnormal_chl_id_1] = 2  # 标注
            # print("叶绿素异常且不再迭代计算的值个数：{0}".format(np.sum(iter_reset == 2)))
            iterx[abnormal_chl_id_1] = iter_max[abnormal_chl_id_1]
            abnormal_chl_id_2 = ((chl == predefine.thresholds().chlbad) & (iter_reset == 0) & (iterx < iter_max)) # 原判断
            chl[abnormal_chl_id_2] = 10
            iter_reset[abnormal_chl_id_2] = 1  # 标注
            # print("叶绿素异常并再次迭代计算的值个数：{0}".format(np.sum(iter_reset == 1)))
            _ = abnormal_chl_id_1 | abnormal_chl_id_2
            _Rrs = ((ltemp - TLg) / t_sol / self.tg_sol / mu0 / self.FoBAR)
            # Rrs[:, :, self.red][_] = _Rrs[:, :, self.red][_]
            # /* Shall we continue iterating */
            # 找出停止迭代的像元

            # /* Shall we continue iterating */,
            abnormal_id_3 = ((np.abs(refl_nir - last_refl_nir) < np.abs(predefine.thresholds().nir_chg * refl_nir)) | (
                    refl_nir < 0.0))  # 已经触发了停止迭代条件的像元索引,这些位置的上一次结果已经是最合理的了，不再迭代；
            loc_temp = abnormal_chl_id_1 | abnormal_id_3
            # print("停止迭代的元素个数：{0}".format(np.sum(~np.isnan(loc_temp))-np.sum(np.isnan(nLw))))
            last_iter[loc_temp] = 1  # 1表示为最后一次迭代已经完成了

            # tLw_final_temp = tLw * 1.  # 暂时假设结果为nlw

            # 上一次迭代中loc_temp处的结果都是可用的，非loc_temp的结果全部赋值为nan
            # tLw_final[~loc_temp] = np.nan  # tLw_final_temp作为输出，继续迭代不能输出的指定为nan
            tLw_pre[~loc_temp] = np.nan  # loc_temp的位置已经触发了停止迭代的条件，表明上一次的计算结果是合理的
            TLg_pre[~loc_temp] = np.nan
            La_pre[~loc_temp] = np.nan  # loc_temp的位置已经触发了停止迭代的条件，表明上一次的计算结果是合理的
            t_sen_pre[~loc_temp] = np.nan
            t_sol_pre[~loc_temp] = np.nan

            # 继续参与迭代的部分，loc_temp处的就不用了，赋值为nan
            tLw[loc_temp] = np.nan  # 这部分继续迭代计算，这个位置要记录下来，后续迭代过程这一位置就可以不参与计算了
            loc_iter[loc_temp] = np.nan  # 这是要继续参与迭代的像元的位置表示，停止迭代的赋值为nan
            TLg[loc_temp] = np.nan
            La[loc_temp] = np.nan
            t_sen[loc_temp] = np.nan
            t_sol[loc_temp] = np.nan

            for ib in range(self.bands.size):
                TLg_final[:, :, ib] = np.nanmean(np.dstack([TLg_final[:, :, ib], TLg_pre[:, :, ib]]), axis=2)
                La_final[:, :, ib] = np.nanmean(np.dstack([La_final[:, :, ib], La_pre[:, :, ib]]), axis=2)
                tLw_final[:, :, ib] = np.nanmean(np.dstack([tLw_final[:, :, ib], tLw_pre[:, :, ib]]), axis=2)
                t_sen_final[:, :, ib] = np.nanmean(np.dstack([t_sen_final[:, :, ib], t_sen_pre[:, :, ib]]), axis=2)
                t_sol_final[:, :, ib] = np.nanmean(np.dstack([t_sol_final[:, :, ib], t_sol_pre[:, :, ib]]), axis=2)

            Lw = tLw / t_sen * self.tg_sol  # 根据tLw的标识，计算下一次迭代需要的参数
            # mu0 = np.cos(np.deg2rad(self.sza)).reshape(self.sza.shape[0], self.sza.shape[1], 1)
            nLw = Lw / t_sol_final / self.tg_sol / mu0 / self.fsol
            brdf_factor = self.fq_brdf_correct(nLw=nLw*1.)
            nLw = nLw * brdf_factor
            # #  /* Compute final Rrs */
            Rrs = nLw / self.Fo_
            Rrs[:, :, self.red][_] = _Rrs[:, :, self.red][_]
            # # /* Compute final self.chl from final nLw (needed for flagging) */
            chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
                                          b520=self.num_520, b555=self.num_555, b670=self.num_670,
                                          sensorid=self.sensorID)

            chl[chl <= 0] = np.nan
            last_refl_nir = refl_nir*1.

            if iter_num == predefine.thresholds().aer_iter_max:
                last_iter[last_iter !=1] = 1
            else:
                iter_num = iter_num + 1
            # 至此，迭代完成

        if self.resize:
            TLg_final = resize.up_sample(data=TLg_final, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                         lon_tar=self.lon_chunk)
            La_final = resize.up_sample(data=La_final, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                        lon_tar=self.lon_chunk)
            t_sen_final = resize.up_sample(data=t_sen_final, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                           lon_tar=self.lon_chunk)
            t_sol_final = resize.up_sample(data=t_sol_final, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                           lon_tar=self.lon_chunk)

        La_final = self.moving_average(array=La_final, window_size=3)
        TLg_final = self.moving_average(array=TLg_final, window_size=3)
        t_sen_final = self.moving_average(array=t_sen_final, window_size=3)
        t_sol_final = self.moving_average(array=t_sol_final, window_size=3)

        Lw_final = (self.Ltemp - TLg_final - La_final) / t_sen_final * self.tg_sol
        nLw_final = Lw_final / t_sol_final / self.tg_sol / mu0 / self.fsol

        #  /* Compute final Rrs */
        Rrs_final = nLw_final / self.Fo_
        # /* Compute final chl from final nLw (needed for flagging) */
        chl_final = get_chl.get_default_chl(rrs=Rrs_final, bands=self.bands, b443=self.num_443, b490=self.num_490,
                                            b520=self.num_520, b555=self.num_555, b670=self.num_670,
                                            sensorid=self.sensorID)
        # brdf_factor = self.fq_brdf_correct(nLw=nLw_final * 1., chl=chl_final * 1.)
        # nLw = nLw * brdf_factor

        return [La_final, TLg_final, t_sen_final, t_sol_final, chl_final]

    # def aerosol_iter_nir(self):
    #     # /* ------------------------------------------------------------------------------------------------------ */
    #     # /* Begin interations for aerosol with corrections for non-zero nLw(NIR)近红外波段不等于0 */
    #     # /* ------------------------------------------------------------------------------------------------------ */
    #     # / *Initialize tLw as surface + aerosol radiance * /
    #     # taua = 0.1
    #     # self.chl = np.full(shape=(self.rows_chunk, self.columns_chunk), fill_value=predefine.thresholds().seed_chl)
    #     ltemp = self.Ltemp * 1.
    #     t_sol = self.t_sol * 1.
    #     t_sen = self.t_sen * 1.
    #     if self.resize:
    #         ltemp = resize.down_sample_aerosol(lt=ltemp, resize=self.resize,
    #                                            band_nirs=self.nirs_num, band_nirl=self.nirl_num)
    #     self.chl = np.full(shape=ltemp[:, :, -1].shape, fill_value=predefine.thresholds().seed_chl)
    #     last_tLw_nir = np.zeros_like(ltemp)
    #     tLw_nir = np.zeros_like(ltemp)
    #     Rrs = np.zeros_like(ltemp)
    #     Rrs[:, :, self.num_555] = predefine.thresholds().seed_green
    #     Rrs[:, :, self.num_670] = predefine.thresholds().seed_red
    #     rows_, columns_ = self.sza.shape[0], self.sza.shape[1]
    #     mu0 = np.cos(np.deg2rad(self.sza)).reshape(rows_, columns_, 1)
    #     self.brdf_factor = np.ones_like(t_sol)
    #
    #     #  /* Initialize iteration loop */
    #     last_iter = np.zeros_like(mu0[:, :, 0])  # 用于标识是否为最后一次迭代，1表示是，0表示否（继续迭代）
    #     iter_num = 0  # 用于计算耀斑的最大迭代次数，wang指出2次就够
    #     iterx = np.zeros_like(last_iter)  # 每个像元的跌打次数
    #     last_refl_nir = np.full_like(last_iter, fill_value=100.)
    #     iter_reset = np.zeros_like(last_iter)
    #     iter_max = np.full_like(last_iter, fill_value=predefine.thresholds().aer_iter_max)
    #     loc_iter = np.full_like(mu0, fill_value=1.)  # 这个矩阵要乘以3维矩阵的，声明为3维
    #     tLw_final = np.full_like(ltemp, fill_value=np.nan)
    #     TLg_final = np.full_like(ltemp, fill_value=np.nan)
    #     La_final = np.full_like(ltemp, fill_value=np.nan)
    #
    #     cslp = 1. / (predefine.thresholds().ctop - predefine.thresholds().cbot)
    #     cint = -cslp * predefine.thresholds().cbot
    #
    #     # while iter_num <= predefine.thresholds().glint_iter_max:
    #     wd_rad = np.deg2rad(self.winddirection)
    #
    #     while np.nanmin(last_iter) == 0:  # last_iter是一个数组，初始值全为0，用于记录每个像元自身是否达到停止迭代条件任何一个像元没有达到停止迭代条件，均需要继续迭代
    #         # print("iteration {0} times".format(iter_num))
    #         iterx = iterx + 1  # 迭代次数，是一个矩阵，记录每个像元的迭代次数（处于迭代过程中的像元）
    #         # 迭代过程中的iter_num也用于记录迭代次数，不过它是一个数，记录的是这个迭代程序运行的次数
    #         # 耀斑福亮度估算, First, the measured Lt(λ) and the wind
    #         # M. Wang and S. Bailey,Correction of sun glint contamination on the SeaWiFS ocean and atmosphere products
    #         # 第一次其实只需要做出近红外波段的耀斑，助后面选出气溶胶模型
    #         mode = 2  # mode表示计算耀斑时的气溶胶光学厚度数据来源，2表示从外部获取，1和0分别表示设置0.1的固定值或者依据wang的估算方法
    #         tLw = ltemp * loc_iter  # /* Initialize tLw as surface + aerosol radiance */
    #         #    loc_iter表示是否参与计算，通过迭代后，部分像元是不需要参与下一次计算的，这部分像元在
    #         # loc_iter中标记为np.nan,之前的代码没有这一关键标识，可能会产生错误或降低计算效率
    #         TLg = getglint.main_exec(iter_num=iter_num, sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
    #                                  taur=self.taur, La=tLw, F0=self.FoBAR, windspeed=self.wind_speed,
    #                                  winddirection=wd_rad, taua=self.taua, mode=mode)
    #         tLw = tLw - TLg
    #
    #         # >>>>>>>>>>>>>>>>>>>>>>>>>>>近红外迭代的关键过程>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #         # /* Adjust for non-zero NIR water-leaving radiances using IOP model */
    #         # 使用IOP模型去估算近红外波段的离水辐亮度，这样就可以获得近红外波段的气溶胶福亮度，再反推各波段的气溶胶辐亮度
    #         rhown_nir = get_rhown_nir.get_rhown_eval(num_443=self.num_443, num_555=self.num_555,
    #                                                  num_670=self.num_670, nirs_num=self.nirs_num,
    #                                                  nirl_num=self.nirl_num, Rrs=Rrs, chl=self.chl, aw=self.aw,
    #                                                  bbw=self.bbw, bands=self.bands, fqfile=self.fqfile,
    #                                                  sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa)
    #         for ib in range(self.nirs_num, self.nirl_num + 1):
    #             tLw_nir[:, :, ib] = (rhown_nir[:, :, ib] / np.pi * self.Fo[ib] * mu0[:, :, 0] *
    #                                  t_sol[:, :, ib] * t_sen[:, :, ib]) / self.brdf_factor[:, :, ib]
    #             #  /* Iteration damping */
    #             tLw_nir[:, :, ib] = (1.0 - predefine.thresholds().df) * tLw_nir[:, :, ib] + (predefine.thresholds().df
    #                                                                                          * last_tLw_nir[:, :, ib])
    #             # /* Ramp-up ?*/
    #             tLw_nir[:, :, ib][(0 < self.chl) & (self.chl < predefine.thresholds().cbot)] = 0.
    #             loc_temp = (predefine.thresholds().cbot < self.chl) & (self.chl < predefine.thresholds().ctop)
    #             tLw_nir[:, :, ib][loc_temp] = tLw_nir[:, :, ib][loc_temp] * (cslp * self.chl + cint)[loc_temp]
    #             tLw[:, :, ib] = tLw[:, :, ib] - tLw_nir[:, :, ib]
    #             del loc_temp
    #         l_nir1 = tLw[:, :, self.nirs_num] - tLw_nir[:, :, self.nirs_num]
    #         l_nir2 = tLw[:, :, self.nirl_num] - tLw_nir[:, :, self.nirl_num]
    #         # aero_out = aerosol_rad.atmos_corr(l_a_nir1=l_nir1, l_a_nir2=l_nir2, lon=self.lon, lat=self.lat,
    #         #                                   F0=self.FoBAR, bands=self.bands, taur=self.Tau_r,
    #         #                                   aerosol_models_info=self.aerosol_lut_info, pressure=self.pressure,
    #         #                                   sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
    #         #                                   winds_peed=self.wind_speed, relative_humidity=self.rh,
    #         #                                   nirl_num=self.nirl_num, nirs_num=self.nirs_num)
    #         aero_out = aerosol_radV2.calculate(l_a_nir1=l_nir1, l_a_nir2=l_nir2, lon=self.lon, lat=self.lat,
    #                                            F0=self.FoBAR, bands=self.bands, taur=self.Tau_r,
    #                                            aerosol_lut_filepath=self.aerosol_lut_info, pressure=self.pressure,
    #                                            sza=self.sza, vza=self.vza, saa=self.saa, vaa=self.vaa,
    #                                            winds_peed=self.wind_speed, relative_humidity=self.rh,
    #                                            nirl_num=self.nirl_num, nirs_num=self.nirs_num)
    #         if aero_out is None:
    #             TLg_final = resize.up_sample(data=TLg, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
    #                                          lon_tar=self.lon_chunk)
    #             La_final = np.zeros_like(TLg_final)
    #             return [La_final, TLg_final]
    #
    #         La, t_sen, t_sol, self.taua, aer1, aer2 = aero_out
    #         # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<近红外迭代的关键过程<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #         # airmass = 1 / np.cos(np.deg2rad(self.sza)) + 1 / np.cos(np.deg2rad(self.vza))
    #         # t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass)
    #         # La_upscale[:, :, self.nirs_num] = La_upscale[:, :, self.nirs_num] * t_o2
    #
    #         tLw = tLw - La
    #         Lw = tLw / t_sen * self.tg_sol
    #         nLw = Lw / t_sol / self.tg_sol / mu0 / self.fsol * self.brdf_factor
    #
    #         # /* Compute new estimated chlorophyll */
    #         refl_nir = Rrs[:, :, self.red]  # 这是前面初始化的一个红光波段反射率
    #         for ib in range(self.nirs_num, self.nirl_num + 1):
    #             last_tLw_nir[:, :, ib] = tLw_nir[:, :, ib]
    #         for ib in range(self.nwvis):
    #             Rrs[:, :, ib] = nLw[:, :, ib] / self.FoBAR[ib]
    #         self.chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
    #                                            b520=self.num_520, b555=self.num_555, b670=self.num_670,
    #                                            sensorid=self.sensorID)
    #
    #         #    if we passed atmospheric correction but the spectral distribution of
    #         # Rrs is bogus (self.chl failed), assume this is a turbid-water case and
    #         # reseed iteration as if all 670 reflectance is from water.
    #
    #         loc_temp = ((self.chl == predefine.thresholds().chlbad) & (iter_reset == 0) & (iterx < iter_max))
    #         self.chl[loc_temp] = 10
    #         Rrs[:, :, self.red][loc_temp] = ((ltemp[:, :, self.red] - TLg[:, :, self.red]
    #                                           ) / t_sol[:, :, self.red] / self.tg_sol[:, :, self.red] /
    #                                          mu0[:, :, 0] / self.FoBAR[self.red])[loc_temp]
    #         iter_reset[loc_temp] = 1
    #         del loc_temp
    #
    #         #     if we already tried a reset, and still no convergence, force one last
    #         # pass with an assumption that all red radiance is water component, and
    #         # force iteration to end.  this will be flagged as atmospheric correction
    #         # failure, but a qualitatively useful retrieval may still result.
    #         loc_temp = ((self.chl == predefine.thresholds().chlbad) & (iter_reset == 1) & (iterx < iter_max))
    #         self.chl[loc_temp] = 10
    #         Rrs[:, :, self.red][loc_temp] = ((ltemp[:, :, self.red] - TLg[:, :, self.red]
    #                                           ) / t_sol[:, :, self.red] / self.tg_sol[:, :, self.red] /
    #                                          mu0[:, :, 0] / self.FoBAR[self.red])[loc_temp]
    #         iterx[loc_temp] = iter_max[loc_temp]
    #         iter_reset[loc_temp] = 2
    #         del loc_temp
    #
    #         # /* Shall we continue iterating */
    #         # 找出停止迭代的像元
    #         # 暂时假设结果为nlw
    #
    #         tLw_final_temp = tLw * 1.
    #
    #         # /* Shall we continue iterating */
    #         if iter_num > predefine.thresholds().aer_iter_max:  # 所有的像元迭代均超过10次了
    #             last_iter = np.ones_like(mu0)
    #             for ib in range(self.bands.size):
    #                 TLg_final[:, :, ib] = np.nansum(np.dstack([TLg_final[:, :, ib], TLg[:, :, ib]]), axis=2)
    #                 La_final[:, :, ib] = np.nansum(np.dstack([La_final[:, :, ib], La[:, :, ib]]), axis=2)
    #                 tLw_final[:, :, ib] = np.nanmean(np.dstack([tLw_final[:, :, ib], tLw_final_temp[:, :, ib]]), axis=2)
    #         else:  # 其它停止迭代的条件和位置
    #             last_iter = np.zeros_like(mu0)
    #             loc_temp = ((np.abs(refl_nir - last_refl_nir) < np.abs(predefine.thresholds().nir_chg * refl_nir)) | (
    #                     refl_nir < 0.0))  # 已经达到了迭代条件的像元索引
    #             last_iter[loc_temp] = 1
    #             tLw_final_temp[~loc_temp] = np.nan  # tLw_final_temp作为输出，继续迭代不能输出的指定为nan
    #             tLw[loc_temp] = np.nan  # 这部分继续迭代计算，这个位置要记录下来，后续迭代过程这一位置就可以不参与计算了
    #             loc_iter[loc_temp] = np.nan  # 这是要继续参与迭代的像元的位置表示，停止迭代的赋值为nan
    #             TLg[~loc_temp] = np.nan
    #             La[~loc_temp] = np.nan
    #             for ib in range(self.bands.size):
    #                 TLg_final[:, :, ib] = np.nansum(np.dstack([TLg_final[:, :, ib], TLg[:, :, ib]]), axis=2)
    #                 La_final[:, :, ib] = np.nansum(np.dstack([La_final[:, :, ib], La[:, :, ib]]), axis=2)
    #                 tLw_final[:, :, ib] = np.nanmean(np.dstack([tLw_final[:, :, ib], tLw_final_temp[:, :, ib]]), axis=2)
    #
    #         Lw = (ltemp - TLg_final - La_final) / t_sen * self.tg_sol
    #         mu0 = np.cos(np.deg2rad(self.sza)).reshape(self.sza.shape[0], self.sza.shape[1], 1)
    #         nLw = Lw / t_sol / self.tg_sol / mu0 / self.fsol
    #         brdf_factor = self.fq_brdf_correct(nLw=nLw)
    #         self.nLw = nLw *brdf_factor
    #         #  /* Compute final Rrs */
    #         Rrs = self.nLw / self.Fo_
    #         # /* Compute final self.chl from final nLw (needed for flagging) */
    #         self.chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
    #                                            b520=self.num_520, b555=self.num_555, b670=self.num_670,
    #                                            sensorid=self.sensorID)
    #         self.chl[self.chl <= 0] = np.nan
    #         last_refl_nir = refl_nir
    #         iter_num = iter_num + 1
    #         if iter_num >= predefine.thresholds().aer_iter_max:
    #             break
    #         # 至此，迭代完成
    #
    #     if self.resize:
    #         TLg_final = resize.up_sample(data=TLg_final, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
    #                                      lon_tar=self.lon_chunk)
    #         La_final = resize.up_sample(data=La_final, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
    #                                     lon_tar=self.lon_chunk)
    #         t_sen = resize.up_sample(data=t_sen, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
    #                                  lon_tar=self.lon_chunk)
    #         t_sol = resize.up_sample(data=t_sol, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
    #                                  lon_tar=self.lon_chunk)
    #     return [La_final, TLg_final, t_sen, t_sol, self.chl]

    def fq_brdf_correct(self, nLw):
        #  /* Compute f/Q correction and apply to nLw */
        Rrs= nLw/self.FoBAR
        chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
                                      b520=self.num_520, b555=self.num_555, b670=self.num_670, sensorid=self.sensorID)
        brdf_mod = brdf.BRDF(vza=self.vza, sza=self.sza, vaa=self.vaa, saa=self.saa, bands=self.bands,
                             F0=self.FoBAR, chl=chl, nlw=nLw, ws=self.wind_speed,
                             b443=self.num_443, b490=self.num_490, b520=self.num_520, b555=self.num_555,
                             b670=self.num_670, foqopt="FOQMOREL", fqfile=self.fqfile, sensorid=self.sensorID)
        brdf_factor = brdf_mod.ocbrdf()
        if self.resize:
            brdf_factor = resize.up_sample(data=brdf_factor, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                           lon_tar=self.lon_chunk)
        return brdf_factor

    def meteor_parameter(self):
        # print("load atmospheric parameters: pressure, O3, NO2, water vapor, reality humidity, wind etc:...")
        tx = atmosphericParameter.get(Lon=self.lon, Lat=self.lat, year=self.year, month=self.month, day=self.day,
                                      time='03:00')
        (self.wind_speed, self.winddirection, self.pressure, self.o3, self.rh, self.water_vapor, self.strat_no2,
         self.trop_no2, self.taua) = (tx)
        del tx
        self.taua = self.taua.reshape((self.wind_speed.shape[0], self.wind_speed.shape[1], 1))
        if self.resize:
            _ = resize.down_sample_sentinel(
                [self.rh, self.pressure, self.water_vapor, self.o3, self.wind_speed, self.winddirection],
                self.resize)
            [self.rh, self.pressure, self.water_vapor, self.o3, self.wind_speed, self.winddirection] = _

    def gas_absorb(self):
        # 3. 瑞利光学厚度的气压校正
        # /* Pressure correct the Rayleigh optical thickness */
        factor = self.pressure / 1013.25
        self.taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * self.Tau_r.reshape(1, 1, -1)
        # 4. 臭氧透过率校正，臭氧从欧洲下载，详情见函数内部。Mobely给的消光截面单位是错误的
        tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=self.sza, vza=self.vza, koz=self.k_oz,
                                                                       concentration=self.o3)
        tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=self.k_no2, sza=self.sza, vza=self.vza,
                                                                          strat_no2=self.strat_no2,
                                                                          trop_no2=self.trop_no2)
        tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=self.t_co2, sza=self.sza, vza=self.vza)
        tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=self.water_vapor, sza=self.sza,
                                                                          vza=self.vza, zia_table=self.Zia_table)
        if self.sensorID in ("olcis3a", "olcis3b"):
            self.tg_sol = tg_solar_o3 * tg_solar_no2
            self.tg_sen = tg_sensor_o3 * tg_sensor_no2
        else:
            self.tg_sol = tg_solar_o3 * tg_solar_no2 * tg_solar_co2 * tg_solar_h2o  # 其它吸收暂时不考虑
            self.tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_sensor_co2 * tg_sensor_h2o
        if self.resize:
            tg_sol = resize.up_sample(data=self.tg_sol, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                      lon_tar=self.lon_chunk)
            tg_sen = resize.up_sample(data=self.tg_sen, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                      lon_tar=self.lon_chunk)
            taur = resize.up_sample(data=self.taur, lat=self.lat, lon=self.lon, lat_tar=self.lat_chunk,
                                    lon_tar=self.lon_chunk)
            return taur, tg_sol, tg_sen
        return self.taur, self.tg_sol, self.tg_sen

    def out_varable(self):
        name_id = os.path.basename(self.infile).split(".")[0]
        outfile = os.path.dirname(self.infile) + os.sep + name_id + "_L2.H5"
        self.ds, self.navi_group, self.geo_group = output.create(outfile=outfile)

    def unit(self, lt):
        """
        Args:
            lt: 辐亮度
        Returns:
            将辐亮度的单位转化为w*m-2*um-1*sr-1,符合seadas文件F0的单位
        """
        if self.sensorID in ["olcis3a", "olcis3b"]:
            lt = lt / 10.
        else:
            lt = lt * 1.
        return lt

    def oxygen_correct(self):
        """
        7.氧气校正：
        """
        if self.sensorID in ["sdgsat1mii", "hy1ccocts", "coctsh1d", "seawifs"]:
            airmass = 1 / np.cos(np.deg2rad(self.sza)) + 1 / np.cos(np.deg2rad(self.vza))
            a_o2 = gas_transmittance.oxygen_ray(airmass)
            t_o2 = gas_transmittance.oxygen_aer(airmass)
        else:
            a_o2 = 1
            t_o2 = 1
        return a_o2, t_o2

    def moving_average(self, array, window_size):
        """
         使用窗口大小为window_size进行平均值平滑
        :param array:
        :param window_size:
        :return:
        """
        # from scipy.ndimage import uniform_filter
        from scipy.ndimage import generic_filter
        kernel = np.ones((3, 3, 1), dtype=np.uint8)
        dilated_array = cv2.dilate(array, kernel, 2)
        # smoothed_array = uniform_filter(array, size=3, mode="nearest")
        # smoothed_array = cv2.blur(dilated_array, ksize=(window_size, window_size), borderType=cv2.BORDER_CONSTANT)
        smoothed_array = generic_filter(dilated_array, np.nanmean, mode="nearest", size=3)
        return smoothed_array


if __name__ == '__main__':
    Calcu().run_main()