# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/19 15:06
@FileName: validation.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
# import glob

import h5py
import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from scipy.optimize import curve_fit
from share import rmse_mape_r2, array_simplify
# import cv2
# import skimage.measure
from scipy.interpolate import interpn
import array_simplify
from scipy import interpolate


def density_scatter(x, y, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x1, y1, z1 = x[idx], y[idx], z[idx]
    return x1, y1, z1


def funcline(x, a, b):
    return a * x + b


class AcValidation(object):
    def __init__(self):

        self.size = 10
        plt.rc('font', family='Times New Roman', weight='bold', size=self.size)
        self.font = {'family': 'Times New Roman',
                     'weight': 'bold',
                     'size': self.size}
        self.cm = 1 / 2.54
        self.dpi = 600
        self.axis_width = 1.5
        self.a4width = 13.8
        self.scale_wh = 1
        self.colors = ["#800000", "#A0522D", "#FF8C00", "#B8860B", "#BDB76B", "#556B2F", "#2F4F4F", "#483D8B",
                       "#8B008B"]
        # self.colors = ["darkviolet", "darkblue", "mediumblue", "darkgreen", "r", "maroon", "dimgrey"]
        # self.wavelength2 = [400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709]
        #                     # , 754,760, 764, 767, 779, 865, 885, 900, 940, 1020]
        self.wavelength2 = [400, 443, 490, 560,  665, 779,  865]
        self.wavelength1 = [401, 438, 495, 553, 657, 776, 854]

    def run_main(self):
        self.h5_sence()

    def h5_sence(self):
        self.path = r"G:\SDGsat\calibration\sea\2023\validation\turbid\test"
        file2 = self.path + os.sep + "S3A_OL_1_EFR____20230326T015501_20230326T015801_20230327T023511_0179_097_060_2340_PS1_O_NT_003.SEN3_L2_seadas.hdf"
        path1 = r"G:\SDGsat\calibration\sea\2023\validation\turbid\test\KX10_MII_20230326_E123.53_N36.50_202300171003_L4B"
        file1 = path1 + os.sep + "KX10_MII_20230326_E123.53_N36.50_202300171003_L4B_ROI_L2_1116.H5"

        self.lat_ref = self.read_h5(file=file2, data_id="navigation_data/latitude")
        self.lon_ref = self.read_h5(file=file2, data_id="navigation_data/longitude")
        self.lat_tar = self.read_h5(file=file1, data_id="navigation_data/latitude")
        self.lon_tar = self.read_h5(file=file1, data_id="navigation_data/longitude")

        para_type_1 = ["tg_sol", "tg_sen", "Lt", "Lr",  "rhos", "tLf"]
        para_type_2 = ["tg_sol", "tg_sen", "Lt", "Lr", "rhos", "tLf"]
        para_type_1 = ["rhos"]
        para_type_2 = ["rhos"]
        data_dict = {}
        for j, para in enumerate(para_type_1):
            for i, b_ in enumerate(self.wavelength1):
                data1 = self.read_h5(file=file1, data_id="geophysical_data/" + para_type_1[j] + "_" + str(b_))
                data2 = self.read_h5(file=file2, data_id="geophysical_data/" + para_type_2[j] + "_" + str(
                    self.wavelength2[i]))

                data1, data2 = self.study_area(d1=data1, d2=data2)
                _ = np.dstack([data1, data2])
                data = array_simplify.delete_nan(_)
                x = data[1, :].reshape(-1)  # dn
                y = data[0, :].reshape(-1)  # lt
                df = pd.DataFrame({"lwk": np.array(x),
                                   "seadas": np.array(y)})
                data_dict[str(b_)] = df
            ou = r"G:\SDGsat\calibration\sea\2023\validation\turbid\test\fig"
            self.fig_scatter(data_dict=data_dict, figname=ou + os.sep + "AC_validation_1116_" + para_type_2[j] + ".png")

    def read_h5(self, file, data_id):
        h5 = h5py.File(file, mode="r")
        print(data_id)
        ds = h5[data_id]
        value = ds[()]*1.
        try:
            Fillvalue = ds.attrs["_FillValue"]
            value[value == Fillvalue[0]] = np.nan
        except:
            pass
        try:
            valid_max = ds.attrs["valid_max"]
            valid_min = ds.attrs["valid_min"]
            value[value < valid_min[0]] = np.nan
            value[value > valid_max[0]] = np.nan
        except:
            pass
        try:
            scale_factor = ds.attrs["scale_factor"]
            value = value * scale_factor[0]
        except:
            pass
        try:
            add_offset = ds.attrs["add_offset"]
            value = value + add_offset[0]
        except:
            pass

        # tho1, tho2 = np.nanpercentile(value, [5, 95])
        # value[value < tho1] = np.nan
        # value[value > tho2] = np.nan
        # print(np.nanmean(value))
        return value

    def study_area(self,d1,d2):

        # 取共同区域
        south_area = np.max([np.nanmin(self.lat_tar), np.nanmin(self.lat_ref)])
        north_area = np.min([np.nanmax(self.lat_tar), np.nanmax(self.lat_ref)])
        west_area = np.max([np.nanmin(self.lon_tar), np.nanmin(self.lon_ref)])
        east_area = np.min([np.nanmax(self.lon_tar), np.nanmax(self.lon_ref)])
        # 目标传感器
        loc1 = np.where((south_area < self.lat_tar) & (self.lat_tar < north_area) &
                        (west_area < self.lon_tar) & (self.lon_tar < east_area))
        if loc1[0].size < 10 or loc1[1].size < 10:
            print("No overlapping areas")
            return 0
        up, low, left, right = np.min(loc1[0]), np.max(loc1[0]), np.min(loc1[1]), np.max(loc1[1])
        d1 = d1[up:low, left:right]
        self.lat_tar = self.lat_tar[up:low, left:right]
        self.lon_tar = self.lon_tar[up:low, left:right]

        # 参考传感器
        loc2 = np.where(
            (south_area < self.lat_ref) & (self.lat_ref < north_area) & (west_area < self.lon_ref) &
            (self.lon_ref < east_area))
        if loc2[0].size < 10 or loc2[1].size < 10:
            print("No overlapping areas")
            return 0
        up2, low2, left2, right2 = np.min(loc2[0]), np.max(loc2[0]), np.min(loc2[1]), np.max(loc2[1])
        d2 = d2[up2:low2, left2:right2]
        self.lat_ref = self.lat_ref[up2:low2, left2:right2]
        self.lon_ref = self.lon_ref[up2:low2, left2:right2]

        d2 = interpolate.griddata((self.lat_ref.flatten(), self.lon_ref.flatten()), d2.flatten(),
                                  (self.lat_tar, self.lon_tar), method='linear')

        return d1, d2

    def fig_scatter(self, data_dict, figname: str):
        nrows = 3
        ncolumns = 3
        single_width = self.a4width / ncolumns
        single_height = single_width * 1 / self.scale_wh
        fig, axes_arr = plt.subplots(nrows=nrows, ncols=ncolumns,
                                     figsize=(ncolumns * single_width * self.cm * 1.01,
                                              nrows * single_height * self.cm * 1.01))  # , subplot_kw={"projection": 'scatter_density'}

        for i in range(nrows * ncolumns):
            ax = axes_arr[i // ncolumns, i % ncolumns]
            if i + 1 > self.wavelength2.__len__():
                ax.axis('off')
                continue
            df = data_dict[str(self.wavelength1[i])]
            df1 = df.sample(n=1000, replace=True, random_state=1)

            x = df1.loc[:, ["lwk"]].to_numpy().reshape(-1)
            y = df1.loc[:, ["seadas"]].to_numpy().reshape(-1)
            min_ = np.min([np.min(x), np.min(y)])
            max_ = np.max([np.max(x), np.max(y)])

            ax.set_ylim(min_ * 0.8, max_ * 1.2)
            ax.set_xlim(min_ * 0.8, max_ * 1.2)

            x_, y_, z_ = density_scatter(x, y)
            f1 = ax.scatter(x_, y_, c=z_, s=0.5, cmap="viridis")
            min_, max_ = ax.get_xlim()
            l2_, = ax.plot([min_, max_], [min_, max_], color="black", linestyle="--", linewidth=self.axis_width)

            p_est, err_est = curve_fit(funcline, x, y)
            performance_index = rmse_mape_r2.evaluation(x, y)
            ybox1 = TextArea(str(int(self.wavelength2[i])) + " nm", textprops=dict(color="black"))
            ybox2 = TextArea("$MR={0}$".format(str(round(performance_index["mr"], 2))), textprops=dict(
                color="darkred", weight="normal", fontproperties={"size": 7}))
            ybox4 = TextArea("$MAPD={0}\,\%$".format(str(round(performance_index["mapd"], 2))), textprops=dict(
                color="darkred", weight="normal", fontproperties={"size": 7}))
            # xbox = HPacker(children=xbox1, align="center", pad=0, sep=5)
            ybox = VPacker(children=[ybox1, ybox2, ybox4], align="left", pad=0, sep=0)
            anchored_text = AnchoredOffsetbox(loc="upper left", child=ybox, pad=1., frameon=False,
                                              bbox_transform=ax.transAxes, borderpad=0.)
            ax.add_artist(anchored_text)

            ax.tick_params(axis='x', which="major", direction='inout', length=10, width=self.axis_width, colors='black',
                           labelrotation=0, labelsize=self.size)
            ax.tick_params(axis='x', which="minor", direction='in', length=3, width=self.axis_width, colors='black',
                           labelrotation=0, labelsize=self.size)
            ax.tick_params(axis='y', which="major", direction='inout', length=10, width=self.axis_width, colors='black',
                           labelrotation=0, labelsize=self.size)
            ax.tick_params(axis='y', which="minor", direction='in', length=3, width=self.axis_width, colors='black',
                           labelrotation=0, labelsize=self.size)

            ax.spines['bottom'].set_linewidth(self.axis_width)  # 设置底部坐标轴的粗细
            ax.spines['left'].set_linewidth(self.axis_width)  # 设置左边坐标轴的粗细
            ax.spines['right'].set_linewidth(self.axis_width)  # 设置右边坐标轴的粗细
            ax.spines['top'].set_linewidth(self.axis_width)  # 设置上部坐标轴的粗细
            # ax.set_aspect(1)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
            # ax.set_xticks([350, 450, 550, 650, 750, 850, 950])
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(5))
            if i == self.wavelength2.__len__() - 1:
                ax.legend([l2_, f1], [u"1:1 line", u""],
                          bbox_to_anchor=(1.2, 0.5), prop=self.font, fontsize=self.size, ncol=1)

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel(u'lwk_drived', fontdict=self.font)
        plt.ylabel(u'seadas_drived', fontdict=self.font)  # labelpad=10,
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.3, hspace=0.15)

        # figname = self.path + os.sep + self.fileID + r'calibrationcoefficient_300dpi.png'
        plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.02, format="jpeg")
        # plt.savefig(figname[0:-10] + '600dpi.png', dpi=600, bbox_inches='tight', pad_inches=0.02, format="jpeg")

        # plt.show()
        plt.close()


if __name__ == '__main__':
    AcValidation().run_main()
