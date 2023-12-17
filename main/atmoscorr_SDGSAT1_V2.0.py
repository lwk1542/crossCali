# -*- coding: utf-8 -*-
"""
@Time    : 2022/11/5 16:47
@FileName: atmoscorr_SDGSAT1_V1.0.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
针对已经裁剪过的影像
"""

from scipy import interpolate
from osgeo import gdal
import skimage.measure
import numpy as np
import datetime
import os
import gc
from utils import read_img_info
from sharepy import preprocessing
from l2gen import atmosphericParameter, gas_transmittance, rayleigh_rad_V201, get_rhown_nir, aerosol_rad, \
    whitecap_rad, get_chl, read_lut, getglint, predefine


class Calcu(object):
    def __init__(self):
        """
        Args:
            filepath:
        """
        # ++++++++++++++++++++++++++++++++++需要设置的参数+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.sensorID = "sdgsat1mii"
        self.sensor_alt = 505
        self.nonzeroNIR = "want_nirLw"
        self.block_size = 50  # 重采样的尺度
        self.block_size_rows = 150  # 一次读取原始影像的行数
        self.rrc_out = True   # 输出瑞利校正
        self.ltoa_out = False
        self.rrs_out = False
        self.chl_out = False
        self.south, self.north, self.west, self.east = 21.9, 22.8, 113, 114.6
        self.filespath = r"G:\SDGsat\calibration\sea\2023\insitu\imagery\forAeronetOC\L4B\test"  # 数据路径
        # self.zipfiles = general.get_filelistv2("KX10_MII_", ".zip", path=filepath, mode="all")
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.dtype = np.float64
        self.taua_upscale = None
        self.new_ds_chl = None
        self.new_ds_rrs = None
        self.new_ds_Lt = None
        self.new_ds_rrc = None
        self.aerosol_lut_info = None
        self.rayleigh_lut_info = None
        self.sensor_info = None
        self.gains = None
        self.infile = None
        self.fsol = None
        self.Tau_r = None
        self.rh = None
        self.winddirection = None
        self.winds_peed = None
        self.lon = None
        self.pressure = None
        self.lat = None
        self.taua = None
        self.Fo_ = None
        self.fqfile = None
        self.FoBAR = None
        self.lat_upscale = None
        self.lon_upscale = None
        self.sza = None
        self.vza = None
        self.saa = None
        self.vaa = None
        self.vaa_upscale = None
        self.saa_upscale = None
        self.vza_upscale = None
        self.sza_upscale = None
        self.Lt = None
        self.Lt_upscale = None
        self.Ltemp = None
        self.t_sol = None
        self.t_sen = None
        self.tg_sen = None
        self.tg_sol = None
        self.taur = None
        self.block_num = None

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
        self.fqfile = r"share" + os.sep + "common" + os.sep + 'morel_fq.h5'

        dirs = os.listdir(self.filespath)
        print("total {} images:".format(dirs.__len__()), dirs)
        for subdir in dirs:
            if ("KX10_MII_" not in subdir) or ("L4B" not in subdir):
                continue
            print("starting processing {}".format(subdir))
            filepath = self.filespath + os.sep + subdir
            infile_temp = filepath + os.sep +subdir +"_ROI_test.tif"
            if not os.path.isfile(infile_temp):
                print("不存在的ROI文件："+subdir)
                continue
            starttime = datetime.datetime.now()
            # 转为WGS84
            self.infile = preprocessing.convert_coordinates(infile_temp)

            # 二. 获取影像相关的信息
            self.get_img_info(self.infile)
            # 三. 建立需要输出的文件数据驱动
            self.out_varable()
            # 迭代器分块处理数据，节约内存开支
            self.rows_org = self.block_size_rows
            for self.block_num, (data, self.lon, self.lat) in enumerate(self.data_Iterator):
                print("=========processing {}-{} lines========".format(self.rows_org * self.block_num,
                                                                       self.rows_org * (1 + self.block_num)))
                self.Lt = (data.transpose(1, 2, 0) * self.gains.reshape((1, 1, -1)) + self.bias.reshape(
                    (1, 1, -1))) / 10
                (self.rows_org, self.columns_org) = self.Lt[:, :, 0].shape
                if self.ltoa_out:
                    for band_temp in range(self.bands.size):
                        self.new_ds_Lt.GetRasterBand(band_temp + 1).WriteArray(
                            (self.Lt[:, :, band_temp] * 100).astype(np.int16), 0, self.rows_org * self.block_num)
                # continue
                self.resampe()
                # 重采样（升尺度）提高计算效率
                (self.rows_upscale, self.columns_upscale) = self.Lt_upscale[:, :, 0].shape
                # 四. 气象数据加载/计算气体吸收
                self.taur, self.tg_sol, self.tg_sen = self.meteor_para_and_gas_absorb()
                # 上下和下行透过率
                self.Ltemp = self.Lt / self.tg_sen / self.tg_sol
                # 卷云和极化校正：没做
                # 五. 白帽反射
                tLf, self.t_sen, self.t_sol = self.tlf()
                self.Ltemp = self.Ltemp - tLf
                # 六.瑞利
                lr = self.lr()
                self.Ltemp = (self.Ltemp - lr)
                if self.rrc_out:
                    # Rrc = np.pi * self.Ltemp / self.Fo_ / np.cos(
                    #     np.deg2rad(self.sza.reshape(self.rows_org, self.columns_org, 1)))
                    Rrc = np.pi * self.Ltemp / self.Fo_/self.tg_sol/self.t_sol / np.cos(
                        np.deg2rad(self.sza.reshape(self.rows_org, self.columns_org, 1)))

                    for band_temp in range(self.bands.size):
                        # self.new_ds_rrc.GetRasterBand(band_temp + 1).WriteArray(
                        #     (Rrc[:, :, band_temp] * 10000).astype(np.int16), 0, self.rows_org * self.block_num)
                        self.new_ds_rrc.GetRasterBand(band_temp + 1).WriteArray(
                            (Rrc[:, :, band_temp]), 0, self.rows_org * self.block_num)
                    del Rrc, band_temp
                    gc.collect()

                # 8.非F/Q brdf校正：不做了
                # 七.耀斑和近红外迭代气溶胶
                if self.rrs_out:
                    Rrs, chl = self.iter_nir()
                    for band_temp in range(self.bands.size):
                        self.new_ds_rrs.GetRasterBand(band_temp + 1).WriteArray(
                            (Rrs[:, :, band_temp] * 10000).astype(np.int16), 0, self.rows_org * self.block_num)
                    if self.chl_out:
                        self.new_ds_chl.GetRasterBand(1).WriteArray((chl * 10000).astype(np.int16), 0,
                                                                    self.rows_org * self.block_num)
                    del Rrs, band_temp, chl
                    gc.collect()

            if self.ltoa_out: self.new_ds_Lt: None
            if self.rrc_out: self.new_ds_rrc: None
            if self.rrs_out: self.new_ds_rrs: None
            if self.chl_out: self.new_ds_chl: None
            endtime = datetime.datetime.now()
            print("====Total time to process this imagery: {} minutes======".format(
                round((endtime - starttime).seconds / 60), 1))

    def resampe(self):
        # stime = datetime.datetime.now()
        self.sza = np.full_like(self.Lt[:, :, 0], fill_value=self.sza)
        self.vza = np.full_like(self.sza, fill_value=self.vza)
        self.saa = np.full_like(self.sza, fill_value=self.saa)
        self.vaa = np.full_like(self.sza, fill_value=self.vaa)
        self.Lt_upscale = skimage.measure.block_reduce(self.Lt, block_size=(self.block_size, self.block_size, 1),
                                                       func=np.nanmean, cval=np.nan, func_kwargs={'dtype': self.dtype})
        self.sza_upscale = skimage.measure.block_reduce(self.sza, block_size=(self.block_size, self.block_size),
                                                        func=np.nanmean, cval=np.nan, func_kwargs={'dtype': self.dtype})
        self.vza_upscale = skimage.measure.block_reduce(self.vza, block_size=(self.block_size, self.block_size),
                                                        func=np.nanmean, cval=np.nan, func_kwargs={'dtype': self.dtype})
        self.saa_upscale = skimage.measure.block_reduce(self.saa, block_size=(self.block_size, self.block_size),
                                                        func=np.nanmean, cval=np.nan, func_kwargs={'dtype': self.dtype})
        self.vaa_upscale = skimage.measure.block_reduce(self.vaa, block_size=(self.block_size, self.block_size),
                                                        func=np.nanmean, cval=np.nan, func_kwargs={'dtype': self.dtype})
        self.lon_upscale = skimage.measure.block_reduce(self.lon, block_size=(self.block_size, self.block_size),
                                                        func=np.nanmean, cval=np.nan, func_kwargs={'dtype': self.dtype})
        self.lat_upscale = skimage.measure.block_reduce(self.lat, block_size=(self.block_size, self.block_size),
                                                        func=np.nanmean, cval=np.nan, func_kwargs={'dtype': self.dtype})

    def get_img_info(self, infile):
        """
        1.读取影像相关的信息
        Returns:
        """
        image_info = read_img_info.get(infile=infile, sensor_id=self.sensorID, block_size=self.block_size_rows)
        (self.sza, self.vza, self.saa, self.vaa, self.gains, self.bias, self.data_Iterator, self.year, self.month,
         self.day, self.num_443, self.num_490, self.num_520, self.num_555, self.num_670, self.nirs_num, self.nirl_num,
         self.nwvis, self.red) = image_info

        # north=self.north, south=self.south, west=self.west, east=self.east
        date = datetime.datetime.strptime(str(self.year) + str(self.month) + str(self.day), "%Y%m%d")
        doy = int(date.strftime("%j"))
        A, B, C, D, E = 1.00014, 0.01671, 0.9856002831, 3.452868, 360.
        self.fsol = 1. / ((A - B * np.cos(2. * np.pi * (C * doy - D) / E) - 0.000014 * np.cos(
            4. * np.pi * (C * doy - D) / E)) ** 2)
        self.FoBAR = self.Fo * self.fsol
        self.Fo_ = self.FoBAR.reshape((1, 1, -1))
        print("correcting coefficient of solar-earth distance: " + str(self.fsol)[0:5])

    def tlf(self):
        # 5. 移除白帽反射
        # 这个白帽计算使用原始数据大小，能够有效提高计算速度，但是其中的taur是未经过气压校正的，影响应该不大
        self.winds_peed = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
                                               self.winds_peed_upscale.flatten(), (self.lat, self.lon),
                                               method='nearest')
        rho_wc = whitecap_rad.calculate(U10=self.winds_peed, bands=self.bands)
        t_sen = np.empty(shape=self.Lt.shape)
        t_sol = np.empty_like(t_sen)
        tLf = np.empty_like(t_sen)
        mu_upscale = np.cos(np.deg2rad(self.sza))
        mu0_upscale = np.cos(np.deg2rad(self.vza))
        for i in range(self.bands.size):
            t_sen[:, :, i] = np.exp(-0.5 * self.Tau_r[i] / mu0_upscale)
            t_sol[:, :, i] = np.exp(-0.5 * self.Tau_r[i] / mu_upscale)
            tLf[:, :, i] = rho_wc[:, :, i] * t_sol[:, :, i] * t_sen[:, :, i] * self.FoBAR[i] * mu_upscale / np.pi
        return tLf, t_sen, t_sol

    def lr(self):
        # 6. 移除瑞利贡献
        # print("computing rayleigh scattering radaince...")
        lr_upscale = rayleigh_rad_V201.rayleigh(raylut_info=self.rayleigh_lut_info, sza=self.sza_upscale,
                                                vza=self.vza_upscale, saa=self.saa_upscale, vaa=self.vaa_upscale,
                                                F0=self.FoBAR, windspeed=self.winds_peed_upscale,
                                                pressure=self.pressure)

        airmass1 = 1 / np.cos(np.deg2rad(self.sza_upscale)) + 1 / np.cos(np.deg2rad(self.vza_upscale))
        # 7.氧气校正：
        a_o2 = gas_transmittance.oxygen_ray(airmass1)
        t_o2 = 1.0 / gas_transmittance.oxygen_aer(airmass1)
        lr_upscale[:, :, self.nirs_num] = lr_upscale[:, :, self.nirs_num] * a_o2  #
        scaleRayleigh = 1.0 - np.exp(-self.sensor_alt / 10)
        lr_upscale = lr_upscale * scaleRayleigh
        lr = np.empty_like(self.Ltemp)
        for band_num in range(self.bands.size):
            lr[:, :, band_num] = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
                                                      lr_upscale[:, :, band_num].flatten(), (self.lat, self.lon),
                                                      method='nearest')
        return lr

    def iter_nir(self):
        # /* ------------------------------------------------------------------------------------------------------ */
        # /* Begin interations for aerosol with corrections for non-zero nLw(NIR)近红外波段不等于0 */
        # /* ------------------------------------------------------------------------------------------------------ */
        # / *Initialize tLw as surface + aerosol radiance * /
        # taua = 0.1
        chl = np.full(shape=(self.rows_upscale, self.columns_upscale), fill_value=predefine.thresholds().seed_chl)
        ltemp = skimage.measure.block_reduce(self.Ltemp, block_size=(self.block_size, self.block_size, 1),
                                             func=np.nanmin, cval=np.nan)
        t_sol_upscale = skimage.measure.block_reduce(self.t_sol, block_size=(self.block_size, self.block_size, 1),
                                                     func=np.nanmin, cval=np.nan)
        t_sen_upscale = skimage.measure.block_reduce(self.t_sen, block_size=(self.block_size, self.block_size, 1),
                                                     func=np.nanmin, cval=np.nan)

        last_tLw_nir = np.zeros_like(ltemp)
        tLw_nir = np.zeros_like(ltemp)
        Rrs = np.zeros_like(ltemp)
        Rrs[:, :, self.num_555] = predefine.thresholds().seed_green
        Rrs[:, :, self.num_670] = predefine.thresholds().seed_red

        mu0_upscale = np.cos(np.deg2rad(self.sza_upscale)).reshape(self.rows_upscale, self.columns_upscale, 1)
        brdf_upscale = np.ones_like(mu0_upscale)

        #  /* Initialize iteration loop */
        last_iter = np.zeros_like(mu0_upscale[:, :, 0])  # 用于标识是否为最后一次迭代，1表示是，0表示否（继续迭代）
        iter_num = 0  # 用于计算耀斑的最大迭代次数，wang指出2次就够
        iterx = np.zeros_like(last_iter)  # 每个像元的跌打次数
        last_refl_nir = np.full_like(last_iter, fill_value=100.)
        iter_reset = np.zeros_like(last_iter)
        iter_max = np.full_like(last_iter, fill_value=predefine.thresholds().aer_iter_max)
        loc_iter = np.full_like(mu0_upscale, fill_value=1.)   # 这个矩阵要乘以3维矩阵的，声明为3维
        tLw_final = np.full_like(ltemp, fill_value=np.nan)
        TLg_final = np.full_like(tLw_final, fill_value=np.nan)
        La_final = np.full_like(tLw_final, fill_value=np.nan)

        cslp = 1. / (predefine.thresholds().ctop - predefine.thresholds().cbot)
        cint = -cslp * predefine.thresholds().cbot

        # while iter_num <= predefine.thresholds().glint_iter_max:
        wd_rad = np.deg2rad(self.winddirection_upscale)

        while last_iter.min() == 0:  # last_iter是一个数组，初始值全为0，用于记录每个像元自身是否达到停止迭代条件任何一个像元没有达到停止迭代条件，均需要继续迭代
            # print("iteration {0} times".format(iter_num))
            iterx = iterx + 1  # 迭代次数，是一个矩阵，记录每个像元的迭代次数（处于迭代过程中的像元）
            # 迭代过程中的iter_num也用于记录迭代次数，不过它是一个数，记录的是这个迭代程序运行的次数
            # 耀斑福亮度估算, First, the measured Lt(λ) and the wind
            # M. Wang and S. Bailey,Correction of sun glint contamination on the SeaWiFS ocean and atmosphere products
            # 第一次其实只需要做出近红外波段的耀斑，助后面选出气溶胶模型
            mode = 2  # mode表示计算耀斑时的气溶胶光学厚度数据来源，2表示从外部获取，1和0分别表示设置0.1的固定值或者依据wang的估算方法
            tLw = ltemp * loc_iter  # /* Initialize tLw as surface + aerosol radiance */
            #    loc_iter表示是否参与计算，通过迭代后，部分像元是不需要参与下一次计算的，这部分像元在
            # loc_iter中标记为np.nan,之前的代码没有这一关键标识，可能会产生错误或降低计算效率
            TLg_upscale = getglint.main_exec(iter_num=iter_num, sza=self.sza_upscale, vza=self.vza_upscale,
                                             saa=self.saa_upscale, vaa=self.vaa_upscale, taur=self.taur, La=tLw,
                                             F0=self.FoBAR, windspeed=self.winds_peed_upscale, winddirection=wd_rad,
                                             taua=self.taua_upscale, mode=mode)
            tLw = tLw - TLg_upscale

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>近红外迭代的关键过程>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # /* Adjust for non-zero NIR water-leaving radiances using IOP model */
            # 使用IOP模型去估算近红外波段的离水辐亮度，这样就可以获得近红外波段的气溶胶福亮度，再反推各波段的气溶胶辐亮度
            rhown_nir = get_rhown_nir.get_rhown_eval(num_443=self.num_443, num_555=self.num_555,
                                                     num_670=self.num_670, nirs_num=self.nirs_num,
                                                     nirl_num=self.nirl_num, Rrs=Rrs, chl=chl, aw=self.aw,
                                                     bbw=self.bbw, bands=self.bands, fqfile=self.fqfile,
                                                     sza=self.sza_upscale, vza=self.vza_upscale,
                                                     saa=self.saa_upscale, vaa=self.vaa_upscale)
            for ib in range(self.nirs_num, self.nirl_num + 1):
                tLw_nir[:, :, ib] = (rhown_nir[:, :, ib] / np.pi * self.Fo[ib] * mu0_upscale[:, :, 0] *
                                     t_sol_upscale[:, :, ib] * t_sen_upscale[:, :, ib]) / brdf_upscale[:, :, 0]
                #  /* Iteration damping */
                tLw_nir[:, :, ib] = (1.0 - predefine.thresholds().df) * tLw_nir[:, :, ib] + (predefine.thresholds().df
                                                                                             * last_tLw_nir[:, :, ib])
                # /* Ramp-up ?*/
                tLw_nir[:, :, ib][(0 < chl) & (chl < predefine.thresholds().cbot)] = 0.
                loc_temp = (predefine.thresholds().cbot < chl) & (chl < predefine.thresholds().ctop)
                tLw_nir[:, :, ib][loc_temp] = tLw_nir[:, :, ib][loc_temp] * (cslp * chl + cint)[loc_temp]
                tLw[:, :, ib] = tLw[:, :, ib] - tLw_nir[:, :, ib]
                del loc_temp
            l_nir1_upscale = tLw[:, :, self.nirs_num] - tLw_nir[:, :, self.nirs_num]
            l_nir2_upscale = tLw[:, :, self.nirl_num] - tLw_nir[:, :, self.nirl_num]
            aero_out = aerosol_rad.atmos_corr(l_a_nir1=l_nir1_upscale, l_a_nir2=l_nir2_upscale, lon=self.lon_upscale,
                                              lat=self.lat_upscale, F0=self.FoBAR, bands=self.bands, taur=self.Tau_r,
                                              aerosol_models_info=self.aerosol_lut_info, pressure=self.pressure,
                                              sza=self.sza_upscale, vza=self.vza_upscale, saa=self.saa_upscale,
                                              vaa=self.vaa_upscale, winds_peed=self.winds_peed_upscale,
                                              relative_humidity=self.rh, nirl_num=self.nirl_num,
                                              nirs_num=self.nirs_num)
            if aero_out is None:
                break
            La_upscale, t_sensor_upscale, t_solar_upscale, self.taua_upscale, aer1_upscale, aer2_upscale = aero_out
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<近红外迭代的关键过程<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            tLw = tLw - La_upscale
            Lw = tLw / t_sensor_upscale * self.tg_sol
            nLw = Lw / t_solar_upscale / self.tg_sol / mu0_upscale / self.fsol * brdf_upscale

            # /* Compute new estimated chlorophyll */
            refl_nir = Rrs[:, :, self.red]  # 这是前面初始化的一个红光波段反射率
            for ib in range(self.nirs_num, self.nirl_num + 1):
                last_tLw_nir[:, :, ib] = tLw_nir[:, :, ib]
            for ib in range(self.nwvis):
                Rrs[:, :, ib] = nLw[:, :, ib] / self.FoBAR[ib]
            chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
                                          b520=self.num_520, b555=self.num_555, b670=self.num_670)

            #    if we passed atmospheric correction but the spectral distribution of
            # Rrs is bogus (chl failed), assume this is a turbid-water case and
            # reseed iteration as if all 670 reflectance is from water.

            loc_temp = ((chl == predefine.thresholds().chlbad) & (iter_reset == 0) & (iterx < iter_max))
            chl[loc_temp] = 10
            Rrs[:, :, self.red][loc_temp] = ((ltemp[:, :, self.red] - TLg_upscale[:, :, self.red]
                                              ) / t_sol_upscale[:, :, self.red] / self.tg_sol[:, :, self.red] /
                                             mu0_upscale[:, :, 0] / self.FoBAR[self.red])[loc_temp]
            iter_reset[loc_temp] = 1
            del loc_temp

            #     if we already tried a reset, and still no convergence, force one last
            # pass with an assumption that all red radiance is water component, and
            # force iteration to end.  this will be flagged as atmospheric correction
            # failure, but a qualitatively useful retrieval may still result.
            loc_temp = ((chl == predefine.thresholds().chlbad) & (iter_reset == 1) & (iterx < iter_max))
            chl[loc_temp] = 10
            Rrs[:, :, self.red][loc_temp] = ((ltemp[:, :, self.red] - TLg_upscale[:, :, self.red]
                                              ) / t_sol_upscale[:, :, self.red] / self.tg_sol[:, :, self.red] /
                                             mu0_upscale[:, :, 0] / self.FoBAR[self.red])[loc_temp]
            iterx[loc_temp] = iter_max[loc_temp]
            iter_reset[loc_temp] = 2
            del loc_temp

            # /* Shall we continue iterating */
            # 找出停止迭代的像元
            # 暂时假设结果为nlw

            tLw_final_temp = tLw * 1.

            # /* Shall we continue iterating */
            if iter_num > predefine.thresholds().aer_iter_max:  # 所有的像元迭代均超过10次了
                last_iter = np.ones_like(mu0_upscale)
                for ib in range(self.bands.size):
                    TLg_final[:, :, ib] = np.nansum(np.dstack([TLg_final[:, :, ib], TLg_upscale[:, :, ib]]), axis=2)
                    La_final[:, :, ib] = np.nansum(np.dstack([La_final[:, :, ib], La_upscale[:, :, ib]]), axis=2)
                    tLw_final[:, :, ib] = np.nanmean(np.dstack([tLw_final[:, :, ib], tLw_final_temp[:, :, ib]]), axis=2)
            else:  # 其它停止迭代的条件和位置
                last_iter = np.zeros_like(mu0_upscale)
                loc_temp = ((np.abs(refl_nir - last_refl_nir) < np.abs(predefine.thresholds().nir_chg * refl_nir)) | (
                        refl_nir < 0.0))  # 已经达到了迭代条件的像元索引
                last_iter[loc_temp] = 1
                tLw_final_temp[~loc_temp] = np.nan  # tLw_final_temp作为输出，继续迭代不能输出的指定为nan
                tLw[loc_temp] = np.nan  # 这部分继续迭代计算，这个位置要记录下来，后续迭代过程这一位置就可以不参与计算了
                loc_iter[loc_temp] = np.nan  # 这是要继续参与迭代的像元的位置表示，停止迭代的赋值为nan
                TLg_upscale[~loc_temp] = np.nan
                La_upscale[~loc_temp] = np.nan
                for ib in range(self.bands.size):
                    TLg_final[:, :, ib] = np.nansum(np.dstack([TLg_final[:, :, ib], TLg_upscale[:, :, ib]]), axis=2)
                    La_final[:, :, ib] = np.nansum(np.dstack([La_final[:, :, ib], La_upscale[:, :, ib]]), axis=2)
                    tLw_final[:, :, ib] = np.nanmean(np.dstack([tLw_final[:, :, ib], tLw_final_temp[:, :, ib]]), axis=2)

            last_refl_nir = refl_nir
            iter_num = iter_num + 1
            if iter_num >= predefine.thresholds().aer_iter_max:
                break
            # 至此，迭代完成
        TLg = np.empty_like(self.Ltemp)
        La = np.empty_like(TLg)
        t_sensor = np.empty_like(TLg)
        t_solar = np.empty_like(TLg)
        for band_num in range(self.bands.size):
            TLg[:, :, band_num] = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
                                                       TLg_final[:, :, band_num].flatten(), (self.lat, self.lon),
                                                       method='nearest')
            La[:, :, band_num] = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
                                                      La_final[:, :, band_num].flatten(), (self.lat, self.lon),
                                                      method='nearest')
            t_sensor[:, :, band_num] = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
                                                            t_sensor_upscale[:, :, band_num].flatten(),
                                                            (self.lat, self.lon),
                                                            method='nearest')
            t_solar[:, :, band_num] = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
                                                           t_solar_upscale[:, :, band_num].flatten(),
                                                           (self.lat, self.lon), method='nearest')
        chl = interpolate.griddata((self.lat_upscale.flatten(), self.lon_upscale.flatten()),
                                   chl.flatten(), (self.lat, self.lon), method='nearest')
        chl[chl <= 0] = np.nan

        Lw = (self.Ltemp - TLg - La) / t_sensor * self.tg_sol
        mu0 = np.cos(np.deg2rad(self.sza)).reshape(self.rows_org, self.columns_org, 1)
        nLw = Lw / t_solar / self.tg_sol / mu0 / self.fsol

        #  /* Compute f/Q correction and apply to nLw */
        # brdf_mod = brdfmodel.BRDF(vza=self.vza, sza=self.sza, vaa=self.vaa, saa=self.saa, bands=self.bands,
        #                           F0=self.FoBAR, chl=chl, nlw=nLw, ws=self.winds_peed,
        #                           b443=self.num_443, b490=self.num_490, b520=self.num_520, b555=self.num_555,
        #                           b670=self.num_670, foqopt="FOQMOREL", fqfile=self.fqfile)
        # brdf = brdf_mod.ocbrdf()
        # nLw = nLw * brdf

        #  /* Compute final Rrs */
        Rrs = nLw / self.Fo_

        # /* Compute final chl from final nLw (needed for flagging) */
        chl = get_chl.get_default_chl(rrs=Rrs, bands=self.bands, b443=self.num_443, b490=self.num_490,
                                      b520=self.num_520, b555=self.num_555, b670=self.num_670)
        return Rrs, chl

    def meteor_para_and_gas_absorb(self):
        # print("load atmospheric parameters: pressure, O3, NO2, water vapor, reality humidity, wind etc:...")
        self.winds_peed_upscale, self.winddirection_upscale, self.pressure, o3, self.rh, water_vapor, strat_no2, trop_no2, self.taua_upscale = \
            atmosphericParameter.get(Lon=self.lon_upscale, Lat=self.lat_upscale, year=self.year, month=self.month,
                                     day=self.day, time='03:00')
        self.taua_upscale = self.taua_upscale.reshape((self.rows_upscale, self.columns_upscale, 1))
        # 3. 瑞利光学厚度的气压校正
        # /* Pressure correct the Rayleigh optical thickness */
        factor = self.pressure / 1013.25
        taur = factor.reshape(factor.shape[0], factor.shape[1], 1) * self.Tau_r.reshape(1, 1, -1)
        # 4. 臭氧透过率校正，臭氧从欧洲下载，详情见函数内部。Mobely给的消光截面单位是错误的
        # print("computing gas absorbing transmittance: o3, no2, co2, h2o...")
        # 对于区域范围小，空间分辨率高的影像，气体吸收计算单个值就行了
        sza_temp = np.array([[np.nanmean(self.sza)]])
        vza_temp = np.array([[np.nanmean(self.vza)]])
        o3_temp = np.array([[np.nanmean(o3)]])
        strat_no2_temp = np.array([[np.nanmean(strat_no2)]])
        trop_no2_temp = np.array([[np.nanmean(trop_no2)]])
        water_vapor_temp = np.array([[np.nanmean(water_vapor)]])
        tg_sensor_o3, tg_solar_o3 = gas_transmittance.transmittance_o3(sza=sza_temp, vza=vza_temp, koz=self.k_oz,
                                                                       concentration=o3_temp)
        tg_sensor_no2, tg_solar_no2 = gas_transmittance.transmittance_NO2(kno2=self.k_no2, sza=sza_temp, vza=vza_temp,
                                                                          strat_no2=strat_no2_temp,
                                                                          trop_no2=trop_no2_temp)
        tg_sensor_co2, tg_solar_co2 = gas_transmittance.transmittance_co2(t_co2=self.t_co2, sza=sza_temp, vza=vza_temp)
        tg_sensor_h2o, tg_solar_h2o = gas_transmittance.transmittance_h2o(water_vapor=water_vapor_temp, sza=sza_temp,
                                                                          vza=vza_temp, zia_table=self.Zia_table)
        tg_sol = tg_solar_o3 * tg_solar_no2 * tg_solar_co2 * tg_solar_h2o  # 其它吸收暂时不考虑
        tg_sen = tg_sensor_o3 * tg_sensor_no2 * tg_sensor_co2 * tg_sensor_h2o
        return taur, tg_sol, tg_sen

    def create_new_tif(self, infile, outfile, bands, datatype=gdal.GDT_Float32):
        dataset = gdal.Open(infile)
        XSize = dataset.RasterXSize  # 网格的X轴像素数量
        YSize = dataset.RasterYSize  # 网格的Y轴像素数量
        geoTransform = dataset.GetGeoTransform()  # 投影转换信息
        projectionInfo = dataset.GetProjection()  # 投影信息
        driver = gdal.GetDriverByName('GTiff')
        new_dataset = driver.Create(outfile, XSize, YSize, bands, datatype)
        new_dataset.SetGeoTransform(geoTransform)
        new_dataset.SetProjection(projectionInfo)
        return new_dataset

    def out_varable(self):
        if self.rrc_out:
            outfile1 = self.infile.replace(".tif", "_Rrc.tif")
            # 文件是否已经生产
            if os.path.exists(outfile1): os.remove(outfile1)
            self.new_ds_rrc = self.create_new_tif(self.infile, outfile1, self.bands.size)
        if self.ltoa_out:
            outfile4 = self.infile.replace(".tif", "_Lt.tif")
            if os.path.exists(outfile4): os.remove(outfile4)
            self.new_ds_Lt = self.create_new_tif(self.infile, outfile4, self.bands.size)
        if self.rrs_out:
            outfile2 = self.infile.replace(".tif", "_IterNIR_Rrs.tif")
            if os.path.exists(outfile2): os.remove(outfile2)
            self.new_ds_rrs = self.create_new_tif(self.infile, outfile2, self.bands.size)
        if self.chl_out:
            outfile3 = self.infile.replace(".tif", "_chla.tif")
            if os.path.exists(outfile3): os.remove(outfile3)
            self.new_ds_chl = self.create_new_tif(self.infile, outfile3, 1)


if __name__ == '__main__':
    Calcu().run_main()
