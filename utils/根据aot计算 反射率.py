import numpy as np


def model_angstrom(aermod_index: np.ndarray = None, band=None):
    """

    :param aermod_index:
    :param band:
    :return:
    """
    angstrom_funcs = aermod_interp_func('wave_lut', target_value='angstrom')
    bands = band.repeat(aermod_index.size).reshape(1, -1)
    points = np.array([bands]).T
    points = points[:, 0, :]
    angstrom_model = np.zeros(shape=(1, aermod_index.size))
    for index in np.unique(aermod_index):
        # 把同一个模型的反演出来
        loc = np.argwhere(aermod_index == index)[:, 1]
        pts = points[loc, :]
        angstrom_model[0, loc.reshape(-1)] = angstrom_funcs[index](pts).reshape(-1)

    return angstrom_model


def model_albedo(aermod_index: np.ndarray = None, band=None):
    """

    :param aermod_index:
    :param band:
    :return:
    """
    albedo_funcs = aermod_interp_func('wave_lut', target_value='albedo')
    bands = band.repeat(aermod_index.size).reshape(1, -1)
    points = np.array([bands]).T
    points = points[:, 0, :]
    albedo_model = np.zeros(shape=(1, aermod_index.size))
    for index in np.unique(aermod_index):
        # 把同一个模型的反演出来
        loc = np.argwhere(aermod_index == index)[:, 1]
        pts = points[loc, :]
        albedo_model[0, loc.reshape(-1)] = albedo_funcs[index](pts).reshape(-1)

    return albedo_model


def model_select_angstrom(angstrom_measure: float = None, relative_humidity=None, band520: int = 520):
    """
    :param angstrom: 实测的angstrom
    :param aermod_index: 预选的气溶胶模型，可以根据湿度预选20个，或者全部的80个。[0-79]shape=(20 or 80,angstrom.size)
    :return:
    """
    # 根据相对湿度选择出20个模型
    rh_up_start, rh_low_start = rh_select_models(relative_humidity=relative_humidity)
    angstrom_model_rh = np.zeros(shape=(20, angstrom_measure.shape[1]))
    for i in range(20):
        if i < 10:
            aermod_index = rh_up_start + i
        else:
            aermod_index = rh_low_start + i - 10
        angstrom_model_rh[i, :] = model_angstrom(aermod_index=aermod_index, band=band520)

    # 剩下四个模型，选择上下两个边界模型
    diff = angstrom_measure - angstrom_model_rh
    diff[np.isnan(diff)] = 999
    diff[diff < 0] = 999
    idx_bracket_low = np.nanargmin(diff, axis=0).reshape(1, -1)
    del diff
    diff = angstrom_measure - angstrom_model_rh
    diff[np.isnan(diff)] = -999
    diff[diff > 0] = -999
    idx_bracket_up = np.nanargmax(diff, axis=0).reshape(1, -1)
    del diff
    ang_mod_low = np.choose(idx_bracket_low, angstrom_model_rh)
    ang_mod_up = np.choose(idx_bracket_up, angstrom_model_rh)

    # 如果epsilonlow和epsilonup同时大于（或者小于）epsilon_measure，则设置选择的模型只有一个，即最靠近epsilon_measure的那个，
    # 如果靠近的是epsilonlow，则 delta此时等于0，如果靠近的是epsilonup，则 delta此时等于1
    ang_mod_up = ang_mod_up.reshape(1, -1)
    ang_mod_low = ang_mod_low.reshape(1, -1)
    ang_mod_up[np.isnan(ang_mod_up)] = 999
    ang_mod_low[np.isnan(ang_mod_low)] = 999

    delta = np.zeros_like(ang_mod_up)
    judge = (ang_mod_low == ang_mod_up)
    delta[~judge] = (angstrom_measure[~judge] - ang_mod_low[~judge]) / (ang_mod_up[~judge] - ang_mod_low[~judge])
    delta = np.nan_to_num(delta, nan=0, posinf=0, neginf=0)
    delta[(ang_mod_low > angstrom_measure) & (ang_mod_up > angstrom_measure)] = 1
    delta[(ang_mod_low < angstrom_measure) & (ang_mod_up < angstrom_measure)] = 0
    delta[ang_mod_low == ang_mod_up] = 0

    # 最终的模型指数
    aer_model_min = np.zeros(shape=(1, angstrom_measure.size))
    aer_model_min[idx_bracket_low < 10] = idx_bracket_low[idx_bracket_low < 10] + rh_up_start[idx_bracket_low < 10]
    aer_model_min[idx_bracket_low >= 10] = idx_bracket_low[idx_bracket_low >= 10] + rh_low_start[
        idx_bracket_low >= 10] - 10

    aer_model_max = np.zeros(shape=(1, sza.size))
    aer_model_max[idx_bracket_up < 10] = idx_bracket_up[idx_bracket_up < 10] + rh_up_start[idx_bracket_up < 10]
    aer_model_max[idx_bracket_up >= 10] = idx_bracket_up[idx_bracket_up >= 10] + rh_low_start[idx_bracket_up >= 10] - 10

    return aer_model_min, aer_model_max, ang_mod_low, ang_mod_up, delta


def fixdaot(aot: np.ndarray = None, relative_humidity=None,
            wave: np.ndarray = np.array([412, 443, 490, 520, 565, 670, 750, 865]), iwnir_s: int = 6, iwnir_l: int = 7):
    """
    根据aot计算气溶胶辐亮度
    aot shape：
          band1   band2   .。。。bandx
    loc1    xx1     xx2          xx3
    loc2    xx4     xx5          xx6
    .
    locn   xx7      xx8          xx9
    /* -
    --------------------------------------------------------------------------------------- */
    /* fixedaot() - compute aerosol reflectance for fixed aot(lambda)                           */                                                                                        */
    /* B. Franz, August 2004.                                                                   */
    /* ---------------------------------------------------------------------------------------- */
    :return:
    """

    """
    /* compute angstrom and use to select bounding models */
    """
    # 找出两波段的位置
    angst_band1 = np.argmin(np.abs(520 - wave))
    angst_band2 = np.argmin(np.abs(865 - wave))
    # 计算angstrom系数
    angstrom = -np.log(aot[angst_band1] / aot[angst_band2]) / np.log(wave[angst_band1] / wave[angst_band2])
    angstrom[aot[:, angst_band2] <= 0] = 0.

    aer_model_min, aer_model_max, ang_mod_low, ang_mod_up, delta = model_select_angstrom(angstrom_measure=angstrom,
                                                                                         relative_humidity=relative_humidity,
                                                                                         band520=520)

    # /* compute factor for SS approximation, set-up for interpolation */
    # # /* get model phase/albedo function for all wavelengths at this geometry for the two models */
    reaa = saa - vaa
    reaa = np.abs(reaa)
    reaa[reaa > 180.] = reaa[reaa > 180.] - 180
    rhoa = np.full(shape=(wave.size, ang_mod_up), fill_value=np.nan)

    for i, band_i in enumerate(wave):
        phase_min = model_phase(aermod_index=aer_model_min, sza=sza, saa=saa, vza=vza, vaa=vaa, band=band_i)
        phase_max = model_phase(aermod_index=aer_model_max, sza=sza, saa=saa, vza=vza, vaa=vaa, band=band_i)
        albedo_min = model_albedo(aermod_index=aer_model_min, band=band_i)
        albedo_max = model_albedo(aermod_index=aer_model_max, band=band_i)
        f_min = albedo_min * phase_min / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
        f_max = albedo_max * phase_max / 4.0 / np.cos(sza * np.pi / 180) / np.cos(vza * np.pi / 180)
        # 如果查找表的波长与观测的aot的波长不是对应的，则需要插值，使用log变换后再插值
        lnf_min = np.log(f_min)
        lnff_max = np.log(f_max)
        # 插值的过程未完成，假设不用插值

        # 单次散射
        rhoas_min = aot[:, i].T * f_min
        rhoas_max = aot[:, i].T * f_max
        if i == iwnir_s:
            rhoas_min_nirs = rhoas_min
            rhoas_max_nirs = rhoas_max
        if i == iwnir_l:
            rhoas_min_nirl = rhoas_min
            rhoas_max_nirl = rhoas_max

        # 多次散射/* compute MS aerosol reflectance in visible bands */
        rhoamin = rhoas_to_rhoa(aermod_index=aer_model_min, rhoas=rhoas_min, sza=sza, vza=vza, saa=saa, vaa=vaa,
                                band=np.array([band]), aeroob=1)
        rhoamax = rhoas_to_rhoa(aermod_index=aer_model_max, rhoas=rhoas_max, sza=sza, vza=vza, saa=saa, vaa=vaa,
                                band=np.array([band]), aeroob=1)
        rhoa[i, :] = (1.0 - delta) * rhoamin + delta * rhoamax

    eps_min = rhoas_min_nirs / rhoas_min_nirl
    eps_max = rhoas_max_nirs / rhoas_max_nirl

    epsnir = (1.0 - delta) * eps_min + delta * eps_max

    return rhoa, epsnir
