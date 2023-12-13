# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: map_figure.py
@time: 2021/3/11 15:12
@desc:
"""

# def draw(infile=None):
#     font = {'family': 'Times New Roman',
#             'color': 'black',
#             'weight': 'normal',
#             'size': 6
#             }
#     plt.rc('font', family='Times New Roman', size=7)
#     geo = r'D:\test\ac2/H1C_OPER_OCT_L2A_20210128T025000_20210302T190047_12528_10_clip.H5'
#     f_new = h5py.File(geo, 'r')
#
#     bands = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_520','Rrs_565', 'Rrs_670']
#     bands_ = ['Rrs412', 'Rrs443', 'Rrs490', 'Rrs520', 'Rrs565', 'Rrs670']
#
#     fig, axes_arr = plt.subplots(2, 3, figsize=(6, 2.5), sharex=False, sharey=False)
#     for i in range(bands.__len__()):
#         value = f_new['/Geophysical Data/' + bands[i]][()]
#         value_ = f_new['/Geophysical Data/' + bands_[i]][()]
#
#         if i < 3:
#             ax = axes_arr[0, i]
#         else:
#             ax = axes_arr[1, i - 4]
#
#         value = value.reshape(-1)
#         value_ = value_.reshape(-1)
#
#         arr=np.empty(shape=(value.shape[0],2))
#         arr[:,0]=value
#         arr[:,1]=value_
#
#         df=pd.DataFrame(arr)
#         if i==0:
#             df = df[(df.iloc[:, 0] > 0.01) & (df.iloc[:, 1] > .01) & (df.iloc[:, 0] < 0.05) & (df.iloc[:, 1] < 0.05)]
#         elif i==1:
#             df = df[(df.iloc[:, 0] > 0.005) & (df.iloc[:, 1] > .005) & (df.iloc[:, 0] < 0.05) & (df.iloc[:, 1] < 0.05)]
#         elif i==2:
#             df = df[(df.iloc[:, 0] > 0.0075) & (df.iloc[:, 1] > .005) & (df.iloc[:, 0] < 0.05) & (df.iloc[:, 1] < 0.05)]
#
#         else:
#             df = df[(df.iloc[:,0]>0.004)&(df.iloc[:,1]>0)&(df.iloc[:,0]<0.02)&(df.iloc[:,1]<0.02)]
#         x=df.iloc[:,0]
#         y=df.iloc[:,1]
#         f1 = ax.scatter(x, y, marker='o', s=0.4, c='', edgecolors='black')
#
#         par = np.polyfit(x, y, 1, full=True)
#
#         slope = par[0][0]
#         intercept = par[0][1]
#         xl = [min(x), max(x)]
#         yl = [slope * xx + intercept for xx in xl]
#
#         # coefficient of determination, plot text
#         variance = np.var(y)
#         residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(x, y)])
#         Rsqr = np.round(1 - residuals / variance, decimals=2)
#         ax.text(.7 * max(x) , .9 * max(y) + .1 * min(y), '$R^2 = %0.2f$' % Rsqr, fontsize=20)
#         ax.set_xlim([0,max(x)])
#         ax.set_ylim([0, max(y)])
#
#         f2 = ax.plot(xl, yl, c='red')
#
#     # plt.subplots_adjust(bottom=0.1, right=0.99, left=.06, top=0.99, wspace=0.2, hspace=0.15)
#     figname = r'D:\test\ac2/ac2_300dpi.png'
#     plt.savefig(figname, dpi=300)
#     plt.show()
#     plt.close()
#
#     f_new.close()

import datetime
import os

# from cartopy import config
# import cartopy.crs as ccrs
import cartopy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

import filesearch


def preprodata(f, dataID):
    # 读取数据，处理其填充值、最大、最小、增益和偏移系数
    data = f[dataID][()]*1.
    if '_FillValue' in f[dataID].attrs.keys():
        data[data == f[dataID].attrs['_FillValue']] = np.nan
    elif 'bad_value_scaled' in f[dataID].attrs.keys():
        data[data == f[dataID].attrs['bad_value_scaled']] = np.nan
    else:
        pass
    if 'valid_min' in f[dataID].attrs.keys():
        data[data < f[dataID].attrs['valid_min']] = np.nan
    if 'valid_max' in f[dataID].attrs.keys():
        data[data > f[dataID].attrs['valid_max']] = np.nan
    if 'scale_factor' in f[dataID].attrs.keys():
        data = data * f[dataID].attrs['scale_factor']
    elif 'Slope' in f[dataID].attrs.keys():
        data = data * f[dataID].attrs['Slope']
    else:
        pass
    if 'add_offset' in f[dataID].attrs.keys():
        data = data + f[dataID].attrs['add_offset']
    elif 'Offset' in f[dataID].attrs.keys():
        data = data + f[dataID].attrs['Offset']
    else:
        pass

    return data


class Drawfig:
    @staticmethod
    def main():
        # # scatterfig
        # files=filesearch.get_filelist("H1B_RICH_OCT_L1B_20101030T020535_20101030T02262818536_10_L2A_WATERS_NIR-AC_v1_df_Rrs-modis.h5", path=r"G:\high_quality\test_reaa_abs")
        # bands1 = ["Rrs_412", "Rrs_443", "Rrs_490", "Rrs_520", "Rrs_565", "Rrs_670", "Rrs_750", "Rrs_865"]
        # bands2 = ["Rrs_412", "Rrs_443", "Rrs_488", "Rrs_531", "Rrs_555", "Rrs_667", "Rrs_748", "Rrs_869"]
        # for file in files:
        #     df = pd.read_hdf(file)
        #     for i in range(bands1.__len__()):
        #         df1 = df[["HY-" + bands1[i],"MODIS-" + bands2[i]]]
        #         df2=df1.dropna(axis=0,how="any")
        #         x=df2.iloc[:,0]
        #         y = df2.iloc[:,1]
        #         Drawfig.scatterfig(x, y,[bands1[i],bands2[i]])

        # # 天顶福亮度散点图：验证
        # file = r"G:\high_quality\test_reaa_abs\validation/H1B_RICH_OCT_L1B_20130407T011209_20130407T01263131252_10_MOD.1KM.L2A.A2013097.0150.061.2017294155905_crossCalibration.h5"
        # f = h5py.File(file, "r")
        # fignameID=os.path.basename(file)[0:-3]
        # wave = [412, 443, 490, 520, 565, 670, 750, 865]
        # band1s = ["DN_" + str(i) for i in wave]
        # band2s = ["Lt_simu_" + str(i) for i in wave]
        # for i in range(wave.__len__()):
        #     df=pd.DataFrame()
        #     b1 = f["Geophysical Data/"+band1s[i]][()].flatten()
        #     b1r = f["Geophysical Data/" + band2s[i]][()].flatten()
        #     df["tar"]=b1
        #     df["ref"] = b1r
        #     df=df.dropna(axis=0, how="any")
        #     df=df.sample(n=5000)
        #     x = df.iloc[:, 0]
        #     y = df.iloc[:,1]
        #     Drawfig.scatterfig(x, y,[wave[i],wave[i]], figname=fignameID+"Ltoa")

        # # 空间分布
        # file = r"G:\high_quality\test_reaa_abs\validation/H1B_RICH_OCT_L1B_20130407T011209_20130407T01263131252_10_MOD.1KM.L2A.A2013097.0150.061.2017294155905_crossCalibration.h5"
        # f = h5py.File(file, "r")
        # figname = "fig/"+os.path.basename(file)[0:-3] + "_spatialDistri.jpeg"
        # (nrows, ncolumns) = f["Geophysical Data/DN_412"][()].shape
        # wave = [412, 443, 490, 520, 565, 670, 750, 865]
        # band1s = ["DN_" + str(i) for i in wave]
        # band2s = ["Lt_simu_" + str(i) for i in wave]
        # data1 = np.full(shape=(nrows, ncolumns, wave.__len__() + 2), fill_value=np.nan)
        # data2 = np.full_like(data1, fill_value=np.nan)
        # for i, band in enumerate(wave):
        #     data1[:, :, i] = f["Geophysical Data/" + band1s[i]][()]
        #     data2[:, :, i] = f["Geophysical Data/" + band2s[i]][()]
        # data1[:, :, -2] = f["Navigation Data/lat"][()]
        # data1[:, :, -1] = f["Navigation Data/lon"][()]
        # data2[:, :, -2] = f["Navigation Data/lat"][()]
        # data2[:, :, -1] = f["Navigation Data/lon"][()]
        # bands = [str(i) + " nm" for i in wave]
        # Drawfig.spatdistrifig(data1=data1, data2=data2, bands=bands, figname=figname)
        # return
        # 用模拟的TOA进行大气校正，验证rrs的精度
        # files = [[
        #              "H1B_RICH_OCT_L1B_20130407T011209_20130407T01263131252_10_MOD.1KM.L2A.A2013097.0150.061.2017294155905_crossCalibration_L2A_WATERS_NIR-AC_v1.h5",
        #              "MOD.1KM.L2A.A2013097.0150.061.2017294155905.hdf"],
        #          [
        #              "H1B_RICH_OCT_L1B_20130407T011209_20130407T01263131252_10_MOD.1KM.L2A.A2013097.0155.061.2017294155909_crossCalibration.h5",
        #              "MOD.1KM.L2A.A2013097.0155.061.2017294155909.hdf"],
        #          [
        #              "H1B_RICH_OCT_L1B_20130407T011209_20130407T01263131252_10_MOD.1KM.L2A.A2013097.0325.061.2017294155901_crossCalibration.h5",
        #              "MOD.1KM.L2A.A2013097.0325.061.2017294155901.hdf"],
        #          [
        #              "H1B_RICH_OCT_L1B_20130407T011209_20130407T01263131252_10_MOD.1KM.L2A.A2013097.0330.061.2017294155907_crossCalibration.h5",
        #              "MOD.1KM.L2A.A2013097.0330.061.2017294155907.hdf"]]
        # bands0 = ["Rrs_412", "Rrs_443", "Rrs_490", "Rrs_520", "Rrs_565", "Rrs_670"]
        # bands1 = ["Rrs_412", "Rrs_443", "Rrs_488", "Rrs_531", "Rrs_555", "Rrs_667"]
        # for file in files:
        #     f0 = h5py.File(r"G:\high_quality\test_reaa_abs\validation"+os.sep+file[0], "r")
        #     f1 = h5py.File(r"G:\high_quality\test_reaa_abs\validation"+os.sep+file[1], "r")
        #     lat0 = f0["navigation_data/latitude"][()]
        #     lon0 = f0["navigation_data/longitude"][()]
        #     lat1 = f1["navigation_data/latitude"][()]
        #     lon1 = f1["navigation_data/longitude"][()]
        #
        #     # 取共同区域
        #     south_area = np.max([np.min(lat1), np.min(lat0)])
        #     north_area = np.min([np.max(lat1), np.max(lat0)])
        #     west_area = np.max([np.min(lon1), np.min(lon0)])
        #     east_area = np.min([np.max(lon1), np.max(lon0)])
        #
        #     loc0 = np.where((south_area < lat0) & (lat0 < north_area) & (west_area < lon0) & (lon0 < east_area))
        #     if loc0[0].size < 10 or loc0[1].size < 10:
        #         continue
        #     up0, low0, left0, right0 = np.min(loc0[0]), np.max(loc0[0]), np.min(loc0[1]), np.max(loc0[1])
        #
        #     loc1 = np.where((south_area < lat1) & (lat1 < north_area) & (west_area < lon1) & (lon1 < east_area))
        #     if loc1[0].size < 10 or loc1[1].size < 10:
        #         continue
        #     up1, low1, left1, right1 = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])
        #
        #     lat0 = lat0[up0:low0, left0:right0]
        #     lon0 = lon0[up0:low0, left0:right0]
        #     lat1 = lat1[up1:low1, left1:right1]
        #     lon1 = lon1[up1:low1, left1:right1]
        #     fignameID = os.path.basename(file[0])[0:-3]
        #
        #     for band0, band1 in zip(bands0, bands1):
        #         geophy0 = f0["geophysical_data"]
        #         x = preprodata(geophy0, band0)
        #         x = x[up0:low0, left0:right0]
        #
        #         geophy1 = f1["geophysical_data"]
        #         y = preprodata(geophy1, band1)
        #         y = y[up1:low1, left1:right1]
        #
        #         x = interpolate.griddata((lat0.flatten(), lon0.flatten()), x.flatten(), (lat1, lon1), method='linear')
        #         df = pd.DataFrame()
        #         b1 = x.flatten()
        #         b1r = y.flatten()
        #         df["tar"] = b1
        #         df["ref"] = b1r
        #         df = df.dropna(axis=0, how="any")
        #         x = df.iloc[:, 0]
        #         y = df.iloc[:, 1]
        #         Drawfig.scatterfig(x, y, [band0, band1], figname=fignameID + "_Ltoa")
        #

        # 比较我的大气校正和陈树果的大气校正
        files = [
            [
                "H1C_OPER_OCT_L1A_20210130T232500_20210130T233000_12577_10_L2A_WATERS_NIR-AC_v1.h5",
                "H1C_OPER_OCT_L2_20210130T232500_20210130T233000_12577_10.h5"],
            [
                "H1C_OPER_OCT_L1A_20210216T235500_20210217T000000_12820_10_L2A_WATERS_NIR-AC_v1.h5",
                "H1C_OPER_OCT_L2_20210216T235500_20210217T000000_12820_10.h5"],
            [
                "H1C_OPER_OCT_L1A_20210224T024500_20210224T025000_12915_10_L2A_WATERS_NIR-AC_v1.h5",
                "H1C_OPER_OCT_L2_20210224T024500_20210224T025000_12915_10.h5"],
            [
                "H1C_OPER_OCT_L1A_20210224T024500_20210224T025000_12922_10_L2A_WATERS_NIR-AC_v1.h5",
                "H1C_OPER_OCT_L2_20210224T024500_20210224T025000_12922_10.h5"],
            [
                "H1C_OPER_OCT_L1A_20210224T225000_20210224T225500_12935_10_L2A_WATERS_NIR-AC_v1.h5",
                "H1C_OPER_OCT_L2_20210224T225000_20210224T225500_12935_10.h5"],
        ]
        bands0 = ["Rrs_412", "Rrs_443", "Rrs_490", "Rrs_520", "Rrs_565", "Rrs_670"]
        bands1 = ["Rrs412", "Rrs443", "Rrs490", "Rrs520", "Rrs565", "Rrs670"]
        for file in files:
            f0 = h5py.File(r"D:\test\ac3\H1C"+os.sep+file[0], "r")
            f1 = h5py.File(r"D:\test\ac3\H1C"+os.sep+file[1], "r")
            lat0 = f0["navigation_data/latitude"][()]
            lon0 = f0["navigation_data/longitude"][()]
            lat1 = f1["Navigation Data/Latitude"][()]
            lon1 = f1["Navigation Data/Longitude"][()]

            # 取共同区域
            south_area = np.max([np.min(lat1), np.min(lat0)])
            north_area = np.min([np.max(lat1), np.max(lat0)])
            west_area = np.max([np.min(lon1), np.min(lon0)])
            east_area = np.min([np.max(lon1), np.max(lon0)])

            loc0 = np.where((south_area < lat0) & (lat0 < north_area) & (west_area < lon0) & (lon0 < east_area))
            if loc0[0].size < 10 or loc0[1].size < 10:
                continue
            up0, low0, left0, right0 = np.min(loc0[0]), np.max(loc0[0]), np.min(loc0[1]), np.max(loc0[1])

            loc1 = np.where((south_area < lat1) & (lat1 < north_area) & (west_area < lon1) & (lon1 < east_area))
            if loc1[0].size < 10 or loc1[1].size < 10:
                continue
            up1, low1, left1, right1 = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])

            lat0 = lat0[up0:low0, left0:right0]
            lon0 = lon0[up0:low0, left0:right0]
            lat1 = lat1[up1:low1, left1:right1]
            lon1 = lon1[up1:low1, left1:right1]
            fignameID = os.path.basename(file[0])[0:-3]

            for band0, band1 in zip(bands0, bands1):
                geophy0 = f0["geophysical_data"]
                x = preprodata(geophy0, band0)
                x = x[up0:low0, left0:right0]

                geophy1 = f1["Geophysical Data"]
                y = preprodata(geophy1, band1)
                y = y[up1:low1, left1:right1]

                x = interpolate.griddata((lat0.flatten(), lon0.flatten()), x.flatten(), (lat1, lon1), method='linear')
                df = pd.DataFrame()
                b1 = x.flatten()
                b1r = y.flatten()
                df["tar"] = b1
                df["ref"] = b1r
                df = df.dropna(axis=0, how="any")
                try:
                    df=df.sample(n=2000)
                except:
                    pass
                x = df.iloc[:, 0]
                y = df.iloc[:, 1]
                Drawfig.scatterfig(x, y, [band0, band1], figname=fignameID + "_Ltoa")

        return

    @staticmethod
    def scatterfig(x, y, band, figname: str = 'HY1B_DL_crossCalibration_Rrs-validation_400dpi.png'):
        from sklearn.metrics import r2_score
        from matplotlib.ticker import MaxNLocator
        from matplotlib.offsetbox import AnchoredText
        size = 4.5
        x = x.to_numpy().reshape(-1, 1)
        y = y.to_numpy().reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        x_ = np.linspace(np.min(x), np.max(x), 50).reshape(-1, 1)
        y_ = reg.predict(x_)

        x = x.reshape(-1, )
        y = y.reshape(-1, )

        xy1 = np.vstack([x, y])
        z = gaussian_kde(xy1)(xy1)
        idx = z.argsort()
        x1, y1, z1 = x[idx], y[idx], z[idx]

        font = {'family': 'Times New Roman',
                'color': 'black',
                'weight': 'normal',
                'size': size
                }
        plt.rc('font', family='Times New Roman', size=size)
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_subplot(111)
        f1 = ax.scatter(x1, y1, c=z1, cmap='viridis', s=0.2, marker='.', facecolors="none")  # rainbow,jet,turbo,viridis
        (lim0, lim1), (lim2, lim3) = ax.get_xlim(), ax.get_ylim()
        min_ = np.min([lim0, lim1, lim2, lim3])
        max_ = np.max([lim0, lim1, lim2, lim3])
        f2 = ax.plot([min_, max_], [min_, max_], color='black', linewidth=0.5, label='1:1 line')
        f3 = ax.plot(x_, y_, color='red', linewidth=1., label='Fitting line', linestyle='dashed')

        # 统计指标
        MAPD = round(np.mean(np.abs((x - y) / y)) * 100, 2)
        r2 = round(r2_score(x, y), 2)
        slope = round(reg.coef_[0][0], 2)
        intercept = round(reg.intercept_[0], 3)

        # print(slope, intercept, MAPD, r2)

        ax.tick_params(axis='x', direction='in', length=2, width=1.5, colors='black', labelrotation=0, labelsize=size)
        ax.tick_params(axis='y', direction='in', length=2, width=1.5, colors='black', labelrotation=0, labelsize=size)

        if intercept < 0:
            text = str(band[0]) + ' nm \n' + 'Unit: $sr^{-1}$ \n' + '$y={slope}x-{intercept}$ \n $MAPD={mapd}\%$ \n $R^2={r2}$'.format(
                slope=str(slope), intercept=str(np.abs(intercept)), mapd=str(MAPD), r2=str(r2))
        else:
            text = str(band[0]) + ' nm \n' + 'Unit: $sr^{-1}$ \n' + '$y={slope}x+{intercept}$ \n $MAPD={mapd}\%$ \n $R^2={r2}$'.format(
                slope=str(slope), intercept=str(np.abs(intercept)), mapd=str(MAPD), r2=str(r2))
        anchored_text = AnchoredText(text, loc='upper left', frameon=False, prop=dict(size=size))
        ax.add_artist(anchored_text)

        ax.set_xlabel(u'WATERS-drived Rrs', loc='center', fontdict=font, fontsize=6, labelpad=0.)
        ax.set_ylabel(u'ChenShuguo-drived Rrs', loc='center', fontdict=font, fontsize=6, labelpad=0.)
        ax.spines['bottom'].set_linewidth(1.)  # 设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.)  # 设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.)  # 设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.)  # 设置上部坐标轴的粗细
        ax.set_aspect(1)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(loc='lower right', fontsize=size, frameon=False)
        # plt.subplots_adjust(bottom=0.01, right=0.99, left=0.01, top=0.99, wspace=0.08, hspace=0.08)
        # plt.margins(0, 0)

        figname_out = "fig/" + figname + "-" + str(band[0]) + "-" + str(band[1]) + '.jpeg'
        figname_out = filesearch.file_check(figname_out)
        plt.savefig(figname_out, dpi=400, bbox_inches='tight', pad_inches=0.01)
        # plt.savefig(figname[0:-10] + '600dpi.png', dpi=600,bbox_inches='tight')
        # plt.show()
        plt.close()


    @staticmethod
    def spatdistrifig(data1: np.ndarray = None, data2: np.ndarray = None, bands: list = None,
                      figname=None):
        """
        (data1: np.ndarray[float] = None, data2: np.ndarray[float] = None, bands: list = None,
                      figname = None)
        Args:
            figname ():
            bands ():
            data1 (): 数据1：包含至少3个图层，最后两个是纬度、经度，前面若干个为不同波段的数据
            data2 (): 数据2：包含至少3个图层，最后两个是纬度、经度，前面若干个为不同波段的数据，如果不指定data2，则返回data1的分布；如果指定data2，则返回两个的分布以及他们的偏差
        Returns:

        """

        from matplotlib.offsetbox import AnchoredText
        size = 12
        # 确定空间范围
        south = 30.1
        north = 37
        east = 125
        west = 119
        loc1 = np.where(
            (south < data1[:, :, -2]) & (data1[:, :, -2] < north) & (west < data1[:, :, -1]) & (east < data1[:, :, -1]))
        up1, low1, left1, right1 = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])
        data1 = data1[up1:low1, left1:right1, :]
        loc2 = np.where(
            (south < data2[:, :, -2]) & (data2[:, :, -2] < north) & (west < data2[:, :, -1]) & (east < data2[:, :, -1]))
        up2, low2, left2, right2 = np.min(loc2[0]), np.max(loc2[0]), np.min(loc2[1]), np.max(loc2[1])
        data2 = data2[up2:low2, left2:right2, :]
        value1 = data1[:, :, 0:-2]
        value1[np.isnan(value1)] = -999
        value1[value1 < 0] = np.nan
        llats, llons = data1[:, :, -2], data1[:, :, -1]
        value2 = data2[:, :, 0:-2]
        value2[np.isnan(value2)] = -999
        value2[value2 < 0] = np.nan
        llats2, llons2 = data2[:, :, -2], data2[:, :, -1]

        # 插值: 根据data1的经纬度插值data2的数据
        for i in range(value1.shape[2] - 2):
            value2[:, :, i] = interpolate.griddata((llats2.flatten(), llons2.flatten()), value2[:, :, i].flatten(),
                                                   (llats, llons), method='linear')
        del i

        value1[np.isnan(value2)] = np.nan
        value2[np.isnan(value1)] = np.nan

        bias = (value2 - value1) * 100 / value1

        plt.rc('font', family='Times New Roman', weight='normal', size=size)
        font = {'family': 'Times New Roman',
                'color': 'black',
                'weight': 'normal',
                'size': size}

        nrows = 3
        ncolumns = value1.shape[2]

        # 确定作图范围
        plot_south = 29.5
        plot_north = 37
        plot_east = 127.1
        plot_west = 122

        fig = plt.figure(figsize=(ncolumns * 1.5, nrows * 1.5 * (plot_north - plot_south) / (plot_east - plot_west)))

        for i in range(ncolumns):
            ax1 = fig.add_subplot(nrows, ncolumns, 1 + i, projection=cartopy.crs.PlateCarree())
            ax1.coastlines('50m', color="grey")
            ax1.add_feature(cartopy.feature.LAND, edgecolor='grey', facecolor='grey')
            ax1.set_extent([plot_west, plot_east, plot_south, plot_north], cartopy.crs.PlateCarree())
            cmap = plt.get_cmap("jet")
            k1 = np.nanpercentile([value1, value2], [2, 98])
            clevs = np.linspace(k1[0], k1[-1], 21)
            cs1 = ax1.contourf(llons, llats, value1[:, :, i], clevs, cmap=cmap, transform=cartopy.crs.PlateCarree())
            anchored_text1 = AnchoredText("Calibrated $L_{TOA}$" + "\n" + bands[i], loc='upper left', frameon=False,
                                          prop=dict(size=12, color="black"))
            ax1.add_artist(anchored_text1)

            # 第二个
            ax2 = fig.add_subplot(nrows, ncolumns, ncolumns + 1 + i, projection=cartopy.crs.PlateCarree())
            ax2.coastlines('50m', color="grey")
            ax2.add_feature(cartopy.feature.LAND, edgecolor='grey', facecolor='grey')
            ax2.set_extent([plot_west, plot_east, plot_south, plot_north], cartopy.crs.PlateCarree())
            cmap = plt.get_cmap("jet")
            cs2 = ax2.contourf(llons, llats, value2[:, :, i], clevs, cmap=cmap)
            anchored_text2 = AnchoredText("Simulated $L_{TOA}$" + "\n" + bands[i], loc='upper left', frameon=False,
                                          prop=dict(size=12, color="black"))
            ax2.add_artist(anchored_text2)

            # 第三个 bias
            ax3 = fig.add_subplot(3, 8, ncolumns * 2 + 1 + i, projection=cartopy.crs.PlateCarree())
            ax3.coastlines('50m', color="grey")
            ax3.add_feature(cartopy.feature.LAND, edgecolor='grey', facecolor='grey')
            ax3.set_extent([plot_west, plot_east, plot_south, plot_north], cartopy.crs.PlateCarree())
            cmap = plt.get_cmap("gist_rainbow")
            k3 = np.nanpercentile(bias, [2, 98])
            clevs = np.linspace(k3[0], k3[-1], 21)
            cs3 = ax3.contourf(llons, llats, bias[:, :, i], clevs, cmap=cmap)
            anchored_text3 = AnchoredText("Bias" + "\n" + bands[i], loc='upper left', frameon=False, prop=dict(size=12, color="black"))
            ax3.add_artist(anchored_text3)
            ax1.set_adjustable('datalim')
            ax2.set_adjustable('datalim')
            ax3.set_adjustable('datalim')

        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.93, top=0.99, wspace=0.02, hspace=0.02)
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        pos3 = ax3.get_position()

        cb_ax12 = fig.add_axes([pos1.x1 + 0.01, pos2.y0 + 0.01, 0.01, pos1.y1 - pos2.y0 - 0.02])
        cb12 = fig.colorbar(cs1, ax=[ax1, ax2], shrink=0.9, cax=cb_ax12, ticks=np.linspace(k1[0], k1[-1], 5))
        labels = [item.get_position()[1] for item in cb12.ax.get_yticklabels()]
        cb12.ax.set_yticklabels([str(round(float(label), 1)) for label in labels])
        cb12.ax.tick_params(length=4, direction='inout', pad=0.01, labelsize=size)
        cb12.set_label('Calibrated or simulated $L_{TOA} (mWcm^{-2}\mu m^{-1}sr^{-1})$', size=size)

        cb_ax3 = fig.add_axes([pos3.x1 + 0.01, pos3.y0 + 0.01, 0.01, pos3.y1 - pos3.y0 - 0.02])
        cb3 = fig.colorbar(cs3, ax=[ax3], shrink=0.9, cax=cb_ax3, ticks=np.linspace(k3[0], k3[-1], 5))
        labels = [item.get_position()[1] for item in cb3.ax.get_yticklabels()]
        cb3.ax.set_yticklabels(["{:.1f}".format(i) for i in labels])
        cb3.ax.tick_params(length=4, direction='inout', pad=0.01, labelsize=size)
        cb3.set_label('Bias (%)', size=size)

        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0, format='png')
        plt.show()
        plt.close()


def preprocessGeodata(f, dataID):
    # 读取数据，处理其填充值、最大、最小、增益和偏移系数
    data = f['/geophysical_data/' + dataID][()].astype(np.float32)
    if '_FillValue' in f['/geophysical_data/' + dataID].attrs:
        data[data == f['/geophysical_data/' + dataID].attrs['_FillValue']] = np.nan
    if 'valid_min' in f['/geophysical_data/' + dataID].attrs:
        data[data < f['/geophysical_data/' + dataID].attrs['valid_min']] = np.nan
    if 'valid_max' in f['/geophysical_data/' + dataID].attrs:
        data[data > f['/geophysical_data/' + dataID].attrs['valid_max']] = np.nan
    if 'scale_factor' in f['/geophysical_data/' + dataID].attrs:
        data = data * f['/geophysical_data/' + dataID].attrs['scale_factor']
    if 'add_offset' in f['/geophysical_data/' + dataID].attrs:
        data = data + f['/geophysical_data/' + dataID].attrs['add_offset']
    return data


def readfile():
    """
    读取对应MODIS和HY的大气校正数据
    Returns:

    """
    file1s = filesearch.get_filelist("L2A_WATERS_NIR-AC_v1.h5", path=r"G:\high_quality\test_reaa_abs")
    for file1 in file1s:
        f1 = h5py.File(file1, "r")
        lat1 = f1["navigation_data/latitude"][()]
        lon1 = f1["navigation_data/longitude"][()]

        date_str = os.path.basename(file1)[17:25]  # [10:17] MOD.1KM.L2A.A2008065.0255.061.2017255040540.hdf
        year, month, doy = date_str[0:4], date_str[4:6], date_str[6:]
        date = datetime.datetime.strptime(date_str, '%Y%m%d')
        date_str2 = date.strftime('%Y%j')

        file2s = filesearch.get_filelist("MOD.1KM.L2A.A", date_str2, ".hdf",
                                         path=r"G:\high_quality\manual_25_40_118_128")
        bands1 = ["Rrs_412", "Rrs_443", "Rrs_490", "Rrs_520", "Rrs_565", "Rrs_670", "Rrs_750", "Rrs_865"]
        bands2 = ["Rrs_412", "Rrs_443", "Rrs_488", "Rrs_531", "Rrs_555", "Rrs_667", "Rrs_748", "Rrs_869"]
        data = pd.DataFrame()
        for file2 in file2s:
            f2 = h5py.File(file2, "r")
            lat2 = f2["navigation_data/latitude"][()]
            lon2 = f2["navigation_data/longitude"][()]

            # 取共同区域
            south_area = np.max([np.min(lat1), np.min(lat2)])
            north_area = np.min([np.max(lat1), np.max(lat2)])
            west_area = np.max([np.min(lon1), np.min(lon2)])
            east_area = np.min([np.max(lon1), np.max(lon2)])

            # 传感器1
            loc1 = np.where((south_area < lat1) & (lat1 < north_area) & (west_area < lon1) & (lon1 < east_area))
            if loc1[0].size < 10 or loc1[1].size < 10:
                continue
            up1, low1, left1, right1 = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])
            lat1_sharearea = lat1[up1:low1, left1:right1]
            lon1_sharearea = lon1[up1:low1, left1:right1]

            # 传感器2
            loc2 = np.where((south_area < lat2) & (lat2 < north_area) & (west_area < lon2) & (lon2 < east_area))
            if loc2[0].size < 10 or loc2[1].size < 10:
                continue
            up2, low2, left2, right2 = np.min(loc2[0]), np.max(loc2[0]), np.min(loc2[1]), np.max(loc2[1])
            lat2_sharearea = lat2[up2:low2, left2:right2]
            lon2_sharearea = lon2[up2:low2, left2:right2]

            # data1 = np.full(shape=(lat1_sharearea.shape[0], lat1_sharearea.shape[2], 8), fill_value=np.nan)
            # data2 = np.full_like(data1, fill_value=np.nan)
            data_temp = pd.DataFrame()
            for num, (band1, band2) in enumerate(zip(bands1, bands2)):
                value1 = f1["geophysical_data/" + band1][()][up1:low1, left1:right1]
                value2 = preprocessGeodata(f2, band2)[up2:low2, left2:right2]
                value2_interp = interpolate.griddata((lat2_sharearea.flatten(), lon2_sharearea.flatten()),
                                                     value2.flatten(), (lat1_sharearea, lon1_sharearea),
                                                     method='linear')

                data_temp["HY-" + band1] = value1.flatten()
                data_temp["MODIS-" + band2] = value2_interp.flatten()
            data_temp = data_temp.dropna(axis=0, how="all")

        data = data.append(data_temp)

        data = data.reset_index(drop=True)
        datasetfile = r'G:\high_quality\test_reaa_abs/' + os.path.basename(file1)[0:-3] + '_df_Rrs-modis.h5'

        dataseth5 = pd.HDFStore(datasetfile, 'w')
        dataseth5['data'] = data
        dataseth5["/data"].attrs["desc"] = "HY和modis对应位置的Rrs"
        dataseth5.close()

    return 'finish'


if __name__ == '__main__':
    # readfile()
    Drawfig.main()
