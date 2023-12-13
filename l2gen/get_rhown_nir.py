"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: get_rhown_nir.py
@time: 2021/5/22 9:08
@desc:
"""

import numpy as np

"""
/* ---------------------------------------------------------------------- */
/* Convert Rrs[0+] to Rrs[0-]                                             */

/* ---------------------------------------------------------------------- */
float above_to_below(float Rrs) {
    return (Rrs / (0.52 + 1.7 * Rrs));
}

/* ---------------------------------------------------------------------- */
/* Convert Rrs[0-] to Rrs[0+]                                             */

/* ---------------------------------------------------------------------- */
float below_to_above(float Rrs) {
    return (Rrs * 0.52 / (1 - 1.7 * Rrs));
}
"""
import brdf


def above_to_below(rrs):
    return rrs / (0.52 + 1.7 * rrs)


def below_to_above(rrs):
    return rrs * 0.52 / (1 - 1.7 * rrs)


def rhown_nir(num_443=None, num_555=None, num_670=None, nirs_num=None, nirl_num=None, Rrs=None, chl=None, aw=None,
              bbw=None, fqfile=None, bands=None, sza=None, vza=None, saa=None, vaa=None):
    reaa = vaa - 180 - saa
    reaa[reaa < -180] = reaa[reaa < -180] + 360
    reaa[reaa > 180] = reaa[reaa > 180] - 360

    wave = bands
    chl_min = 0.2
    chl_max = 30.

    ib2 = num_443
    ib5 = num_555
    ib6 = num_670

    Rrs2 = Rrs[:, :, ib2]
    Rrs5 = Rrs[:, :, ib5]
    Rrs6 = Rrs[:, :, ib6]

    chl[chl < chl_min] = chl_min
    chl[chl > chl_max] = chl_max

    # // NOMAD fit of apg670 to chl
    apg6 = np.exp(np.log(chl) * 0.9389 - 3.7589)
    apg6[apg6 < 0] = 0
    apg6[apg6 > 0.5] = 0.5

    # /* Compute total absorption at 670 */
    aw6 = aw[ib6]
    a6 = aw6 + apg6

    # /* Go below... */
    Rrs2 = above_to_below(Rrs2)
    Rrs5 = above_to_below(Rrs5)
    Rrs6 = above_to_below(Rrs6)

    # // foqint_morel(wave, nwave, 0.0, 0.0, 0.0, chl_in, foq);
    foq = brdf.foqint_morel(fqfile=fqfile, wave=bands, sza=sza, vzap=vza, reaa=reaa, chl=chl)

    # /* Compute the backscatter slope ala Lee */
    eta = np.zeros_like(Rrs2)
    loc_temp = (Rrs5 > 0) & (Rrs2 > 0)
    eta[loc_temp] = 2. * (1 - 1.2 * np.exp(-0.9 * (Rrs2[loc_temp] / Rrs5[loc_temp])))
    eta[eta < 0] = 0.
    eta[eta > 1] = 1.

    # /* Compute total backscatter at 670 */
    Rrs6_star = Rrs6 / foq[:, :, ib6]
    bbp6 = (Rrs6_star * a6 / (1. - Rrs6_star)) - bbw[ib6]

    # /* Compute normalized water-leaving reflectance at each NIR wavelength */
    rhown = np.zeros(shape=(Rrs2.shape[0], Rrs2.shape[1], wave.size))
    for ib in [nirs_num, nirl_num]:
        if ib == ib6:
            a = a6
        else:
            a = aw[ib]

        # /* Translate bb to NIR wavelength */
        bb = bbp6 * np.power((wave[ib6] / wave[ib]), eta) + bbw[ib]

        # / *Remote - sensing reflectance * /
        salbedo = bb / (a + bb)
        Rrs_nir = foq[:, :, ib6] * salbedo
        # / *Normalized water - leaving reflectance * /
        Rrs_nir = below_to_above(Rrs_nir)
        rhown[:, :, ib] = np.pi * Rrs_nir
        rhown[:, :, ib][Rrs6 <= 0] = 0.0

    return rhown


def get_rhown_eval(num_443=None, num_555=None, num_670=None, fqfile=None, Rrs=None, bands=None, nirs_num=None,
                   nirl_num=None, aw=None, bbw=None, chl=None, sza=None, vza=None, saa=None, vaa=None):
    # seed_chl = 0.0
    # seed_green = 0.0
    # seed_red = 0.0

    # if (want_nirLw || want_nirRrs) {
    #         for (ib = 0; ib < nwave; ib++) {
    #             last_tLw_nir[ib] = 0.0;
    #             tLw_nir[ib] = 0.0;
    #             Rrs[ib] = 0.0;
    #         }
    #         Rrs[green] = seed_green;
    #         Rrs[red ] = seed_red;
    #     }

    rhown = rhown_nir(num_443=num_443, num_555=num_555, num_670=num_670, nirs_num=nirs_num, nirl_num=nirl_num, Rrs=Rrs,
                      chl=chl, aw=aw, bbw=bbw, fqfile=fqfile, bands=bands, sza=sza, vza=vza, saa=saa, vaa=vaa)

    return rhown
