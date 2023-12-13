# -*- coding: utf-8 -*-
import datetime

from scipy import interpolate
import numpy as np

"""
t(θv, λ)ρwc(λ) = [ρwc(λ)]N t(θs, λ) t(θv, λ) 
https://oceancolor.gsfc.nasa.gov/docs/ocssw/whitecaps_8c_source.html#l00037
"""


def calculate(U10=None, bands=None):
    """

    Args:
        U10 (): 海拔10米处风速
        dtrans ():
        bands ():

    Returns:
        whitecaps() - whitecap 反射率

    """
    # stime = datetime.datetime.now()
    wavelength = np.array([412, 443, 490, 510, 555, 670, 765, 865])
    a_wc = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.889, 0.760, 0.645])
    a_wc_inter = interpolate.interp1d(wavelength, a_wc, bounds_error=False, fill_value="extrapolate", kind="linear")(bands)

    rho_wc_N=np.zeros(shape=(U10.shape[0],U10.shape[1],a_wc_inter.size))
    # Stramska and Petelski (2003) wc fractional coverage for underdeveloped seas
    for i in range(bands.size):
        rho_wc_N[U10>6.33,i]=a_wc_inter[i]*1.925*1e-05*(U10[U10>6.33]-6.33)**3
    #rho_wc=rho_wc_N*dtrans
    # https://oceancolor.gsfc.nasa.gov/docs/ocssw/whitecaps_8c_source.html在这个代码里面，白帽没有加漫射透过率,可能是估算的时候已经考虑满射透过率了
    # etime = datetime.datetime.now()
    # print("====Total time for whitecap===: {} minutes======".format(((etime - stime).seconds / 60)))
    return rho_wc_N