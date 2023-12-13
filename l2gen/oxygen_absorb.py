# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/30 15:19
@FileName: oxygen_absorb.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""


# /* ------------------------------------------------------------------- */
# /* correction factor to remove oxygen absorption from La(765)          */
#
# /* ------------------------------------------------------------------- */
# float oxygen_aer(float airmass) {
#     /* base case: m80_t50_strato  visibility (550nm ,0-2km):25km */
#     static float a[] = {-1.0796, 9.0481e-2, -6.8452e-3};
#     return (1.0 + pow(10.0, a[0] + airmass * a[1] + airmass * airmass * a[2]));
# }
def oxygen_aer(airmass):
    a=[-1.0796, 9.0481e-2, -6.8452e-3]
    return 1.0 + pow(10.0, a[0] + airmass * a[1] + airmass * airmass * a[2])

# /* ------------------------------------------------------------------- */
# /* correction factor to replace oxygen absorption to Lr(765)           */
#
# /* ------------------------------------------------------------------- */
# float oxygen_ray(float airmass) {
#     /* base case is the 1976 Standard atmosphere without aerosols */
#     static float a[] = {-1.3491, 0.1155, -7.0218e-3};
#     return (1.0 / (1.0 + pow(10.0, a[0] + airmass * a[1] + airmass * airmass * a[2])));
# }


def oxygen_ray(airmass):
    a = [-1.3491, 0.1155, -7.0218e-3]
    return 1.0 / (1.0 + pow(10.0, a[0] + airmass * a[1] + airmass * airmass * a[2]))