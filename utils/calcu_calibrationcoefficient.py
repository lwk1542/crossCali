# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: H1A_calcu_calibrationcoefficient.py
@time: 2021/7/6 8:22
@desc:
"""
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

from atmosphericCorrection.oceancolorACnirV2.share.seawifs.l2gen import general


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
    y_pre=reg.predict(x)

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
    # f2 = ax.plot([min_, max_], [min_, max_], color='black', linewidth=0.5, label='1:1 line')
    f3 = ax.plot(x_, y_, color='red', linewidth=1., label='Fitting line', linestyle='dashed')

    # 统计指标
    MAPD = round(np.mean(np.abs((y_pre - y) / y)) * 100, 2)

    r2 = round(r2_score(y_pre, y, multioutput='variance_weighted'), 2)
    slope = round(reg.coef_[0][0], 6)
    intercept = round(reg.intercept_[0], 6)

    # print(slope, intercept, MAPD, r2)

    ax.tick_params(axis='x', direction='in', length=2, width=1.5, colors='black', labelrotation=0, labelsize=size)
    ax.tick_params(axis='y', direction='in', length=2, width=1.5, colors='black', labelrotation=0, labelsize=size)

    if intercept < 0:
        text = str(band[
                       0]) + ' nm \n' + 'Unit: $sr^{-1}$ \n' + '$y={slope}x-{intercept}$ \n $MAPD={mapd}\%$ \n $R^2={r2}$'.format(
            slope=str(slope), intercept=str(np.abs(intercept)), mapd=str(MAPD), r2=str(r2))
    else:
        text = str(band[
                       0]) + ' nm \n' + 'Unit: $sr^{-1}$ \n' + '$y={slope}x+{intercept}$ \n $MAPD={mapd}\%$ \n $R^2={r2}$'.format(
            slope=str(slope), intercept=str(np.abs(intercept)), mapd=str(MAPD), r2=str(r2))
    anchored_text = AnchoredText(text, loc='upper left', frameon=False, prop=dict(size=size))
    ax.add_artist(anchored_text)

    ax.set_xlabel(u'DN value', loc='center', fontdict=font, fontsize=6, labelpad=0.)
    ax.set_ylabel(u'Simumlated radiance', loc='center', fontdict=font, fontsize=6, labelpad=0.)
    ax.spines['bottom'].set_linewidth(1.)  # 设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.)  # 设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.)  # 设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.)  # 设置上部坐标轴的粗细
    # ax.set_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc='lower right', fontsize=size, frameon=False)
    # plt.subplots_adjust(bottom=0.01, right=0.99, left=0.01, top=0.99, wspace=0.08, hspace=0.08)
    # plt.margins(0, 0)

    figname_out = "fig/" + figname + "-" + str(band[0]) + "-" + str(band[1]) + '.jpeg'
    figname_out = general.file_check(figname_out)
    plt.savefig(figname_out, dpi=400, bbox_inches='tight', pad_inches=0.01)
    # plt.savefig(figname[0:-10] + '600dpi.png', dpi=600,bbox_inches='tight')
    # plt.show()
    plt.close()
    return reg.coef_[0][0], reg.intercept_[0], MAPD, r2


def exec_func():
    # 天顶福亮度散点图：验证：定标系数
    path = r"G:\high_quality\H1A_26images\calibration_h1a\OCT"
    files = glob.glob(path + os.sep + "*_crossCalibration.h5")
    coefficient = pd.DataFrame(columns=["filename", "date", "gain", "offset", "mapd", "r2"])
    for file in files:
        f = h5py.File(file, "r")
        fignameID = os.path.basename(file)[0:-3]
        wave = [412, 443, 490, 520, 565, 670, 750, 865]
        band1s = ["DN_" + str(i) for i in wave]
        band2s = ["Lt_simu_" + str(i) for i in wave]
        for i in range(wave.__len__()):
            df = pd.DataFrame()
            b1 = f["Geophysical Data/" + band1s[i]][()].flatten()
            b1r = f["Geophysical Data/" + band2s[i]][()].flatten()
            df["tar"] = b1
            df["ref"] = b1r
            df = df.dropna(axis=0, how="any")
            df = df[df["tar"] < 700]
            if df.shape[0] < 100:
                continue
            try:
                df = df.sample(n=5000)
            except:
                pass
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            slope, intercept, MAPD, r2 = scatterfig(x, y, [wave[i], wave[i]], figname=fignameID + "Ltoa")
            date = os.path.basename(file)[17:25]
            coefficient_temp = pd.DataFrame([[os.path.basename(file), date, slope, intercept, MAPD, r2]],
                                            columns=["filename", "date", "gain", "offset", "mapd", "r2"])
            coefficient=coefficient.append(coefficient_temp, ignore_index=True)
        coefficient.to_csv(r"G:\high_quality\H1A_26images\calibration_h1a\OCT\calibration/h1b_calibration_coefficients.csv", header=0)
    return


if __name__ == "__main__":
    exec_func()
