# -*- coding: utf-8 -*-
"""
@author: li wenkai
@contact: lwk1542@hotmail.com
@software: pycharm anaconda python3.7 
@file: brdf.py
@time: 2021/6/30 10:05
@desc:
"""

"""
def ocbrdf(vza: np.ndarray[float] = None, sza: np.ndarray[float] = None, vaa: np.ndarray[float] = None,
    saa: np.ndarray[float] = None, bands: np.ndarray[float] = None, F0: np.ndarray[float] = None,
    chl: np.ndarray[float] = None, nlw: np.ndarray[float] = None, b443:int = None, b490:int = None,
    b520: int = None, b555:int = None, b670:int = None, foqopt:str = "FOQMOREL",
    fqfile = r'C:\git_repository\liwenkai\atmosphericCorrection\LUT/morel_fq.h5'):


    reaa = vaa - 180 - saa
    reaa[reaa < -180] = reaa[reaa < -180] + 360
    reaa[reaa > 180] = reaa[reaa > 180] - 360

    #1 初始化
    brdf=np.ones(shape=(sza.shape[0],sza.shape[1],bands.size))

    # 2 /* transmittance of view path through air & sea interface */
    tf = fresnel_sen(senz=sza, return_tf=0)

    # 3 /* transmittance of solar path through air & sea interface */
    temp=fresnel_sol(bands=bands, solz=sza, ws=ws, return_tf=0)

    brdf=brdf*tf*temp

    #  4 /* Morel f/Q correction */
    foq_morel(senz = vza, solz = sza, vaa=vaa,
              saa= saa, bands= bands, F0= F0,
              chl= chl, nlw= nlw, b443=b443, b490=b490, b520=None,b555int=None,b670=None,
              foqopt="FOQMOREL", fqfile=fqfile)

    #  5 /* Gordon correction of diffuse transmittance */
    dtran_brdf(l2rec, ip, wave, nwave, Fo, nLw, chl, temp)

    return


def fresnel_sen(senz=None, return_tf=0):
    # /* ---------------------------------------------------------------------------- */
    # /* fresnel_sen() - effects of the air-sea transmittance for sensor view         */
    # /*                                                                              */
    # /* Description:                                                                 */
    # /*   This computes effects of the air-sea transmittance (depending on sensor    */
    # /*   zenith angle) on the derived normalized water-leaving radiance.            */
    # /*   Menghua Wang 5/27/02.                                                      */
    # /*                                                                              */
    # /* modified to return fresnel transmittance as option, December 2008, BAF       */
    #
    # /* ---------------------------------------------------------------------------- */
    tf0 = 0.9795218
    nw = 1.334
    mu = np.cos(senz*np.pi / 180)

    sq = np.sqrt(nw * nw - 1. + mu * mu)
    r2 = np.power((mu - sq) / (mu + sq), 2)
    q1 = (1. - mu * mu - mu * sq) / (1. - mu * mu + mu * sq)
    fres = r2 * (q1 * q1 + 1.) / 2.0
    tf = 1. - fres
    brdf = tf0 / tf

    if return_tf != 0:
        return tf
    else:
        return brdf


def fresnel_sol(bands=None,solz=None, ws=None, return_tf=0):
    # /* ---------------------------------------------------------------------------- */
    # /* fresnel_sol() - effects of the air-sea transmittance for solar path          */
    # /*                                                                              */
    # /* Description:                                                                 */
    # /*   This computes the correction factor on normalized water-leaving radiance   */
    # /*   to account for the solar zenith angle effects on the downward irradiance   */
    # /*   from above the ocean surface to below the surface.                         */
    # /*   Menghua Wang 9/27/04.                                                      */
    # /*                                                                              */
    # /* Added windspeed dependence, December 2004, BAF                               */
    # /* Modified to return air-sea transmittance as option, December 2008, BAF       */
    #
    # /* ---------------------------------------------------------------------------- */
    twave= np.array([412., 443., 490., 510., 555., 670.])
    tsigma = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    # /* M Wang, personal communication, red-nir iterpolated */
    tf0_w = [412., 443., 490., 510., 555., 670., 765., 865.]
    tf0_v = [0.965980, 0.968320, 0.971040, 0.971860, 0.973450, 0.977513, 0.980870, 0.984403]

    ws_inter = [0.0,1.9,7.5,16.9,30]
    # c在不同风速下的abcd值矩阵
    c=np.array([
        [
            [-0.0087, -0.0122, -0.0156, -0.0163, -0.0172, -0.0172],
            [0.0638, 0.0415, 0.0188, 0.0133, 0.0048, -0.0003],
            [-0.0379, -0.0780, -0.1156, -0.1244, -0.1368, -0.1430],
            [-0.0311, -0.0427, -0.0511, -0.0523, -0.0526, -0.0478],
        ],
        [
            [-0.0011, -0.0037, -0.0068, -0.0077, -0.0090, -0.0106],
            [0.0926, 0.0746, 0.0534, 0.0473, 0.0368, 0.0237],
            [-5.3E-4, -0.0371, -0.0762, -0.0869, -0.1048, -0.1260],
            [-0.0205, -0.0325, -0.0438, -0.0465, -0.0506, -0.0541],
        ],
        [
            [6.8E-5, -0.0018, -0.0011, -0.0012, -0.0015, -0.0013],
            [0.1150, 0.1115, 0.1075, 0.1064, 0.1044, 0.1029],
            [0.0649, 0.0379, 0.0342, 0.0301, 0.0232, 0.0158],
            [0.0065, -0.0039, -0.0036, -0.0047, -0.0062, -0.0072],
        ],
        [
            [-0.0088, -0.0097, -0.0104, -0.0106, -0.0110, -0.0111],
            [0.0697, 0.0678, 0.0657, 0.0651, 0.0640, 0.0637],
            [0.0424, 0.0328, 0.0233, 0.0208, 0.0166, 0.0125],
            [0.0047, 0.0013, -0.0016, -0.0022, -0.0031, -0.0036],
        ],
        [
            [-0.0081, -0.0089, -0.0096, -0.0098, -0.0101, -0.0104],
            [0.0482, 0.0466, 0.0450, 0.0444, 0.0439, 0.0434],
            [0.0290, 0.0220, 0.0150, 0.0131, 0.0103, 0.0070],
            [0.0029, 0.0004, -0.0017, -0.0022, -0.0029, -0.0033],
        ]
    ])
    ws[ws<0]=0
    sigma = 0.0731 * np.sqrt(ws)
    solz[solz>80]=80
    x = np.log(np.cos(solz*np.pi/180))
    x2 = x * x
    x3 = x * x2
    x4 = x * x3

    c_interp = interpolate.interpn((tsigma.reshape(-1), twave.reshape(-1)), c, np.stack([sigma, bands], axis=2),
                                 method='linear', fill_value=False)
    brdf=1.+c_interp[0]*x+c_interp[1]*x2+c_interp[2]*x3+c_interp[3]*x4

    if return_tf != 0:
        func=interpolate.interp1d(tf0_w,tf0_v,kind ='linear',fill_value="extrapolate",bounds_error=False)
        tf0=func(bands)
        brdf = tf0 / brdf
    return brdf


def foqint_morel(fqfile=r'C:\git_repository\liwenkai\atmosphericCorrection\LUT/morel_fq.h5', wave=None,
                 sza=None, vzap: np.ndarray[float]=None, reaa=None, chl=None):

    fq=h5py.File(fqfile,'r')
    lchl_lut=np.log(fq['chl'][()])
    phi_lut = fq['phi'][()]
    senz_lut = fq['senz'][()]
    solz_lut = fq['solz'][()]
    wave_lut = fq['wave'][()]
    foq_lut = fq['foq'][()]

    chl[chl<0.01]=0.01
    lchl=np.log(chl)
    vzap[vzap<np.nanmin(senz_lut)]=np.nanmin(senz_lut)
    vzap[vzap > np.nanmax(senz_lut)] = np.nanmax(senz_lut)
    fq = interpolate.interpn(
        (wave_lut.reshape(-1), solz_lut.reshape(-1), lchl_lut.reshape(-1), senz_lut.reshape(-1), phi_lut.reshape(-1)), foq_lut,
        np.stack([wave, sza, lchl,vzap,reaa], axis=2), method='linear',bounds_error=False, fill_value=None)

    return fq


def foq_morel(senz: np.ndarray[float] = None, solz: np.ndarray[float] = None, vaa: np.ndarray[float] = None,
              saa: np.ndarray[float] = None, bands: np.ndarray[float] = None, F0: np.ndarray[float] = None,
              chl: np.ndarray[float] = None, nlw: np.ndarray[float] = None, b443:int=None, b490:int=None,
              b520:int=None,b555:int=None,b670:int=None, foqopt:str="QMOREL",
              fqfile=r'C:\git_repository\liwenkai\atmosphericCorrection\LUT/morel_fq.h5'):   
    reaa = vaa - 180 - saa
    reaa[reaa < -180] = reaa[reaa < -180] + 360
    reaa[reaa > 180] = reaa[reaa > 180] - 360
    phip=np.abs(reaa)
    nw=1.334
    senzp = np.arcsin(np.sin(senz *np.pi/180.) / nw) * 180./np.pi
    rrs=nlw/F0
    # 1. /* Compute starting chlorophyll (if not supplied) */
    chl=get_chl.get_default_chl(rrs=rrs, b443=b443, b490=b490, b520=b520, b555=b555, b670=b670)

    # 2. mask 可能出现叶绿素小于0 的情况，这种需要掩膜掉
    mask=chl*1.
    mask[np.isnan(mask)]==-999
    mask[mask<0] = np.nan
    mask2=chl*1.

    # 3.迭代计算brdf和叶绿素
    maxiter = predefine.thresholds().brdf_maxiter
    compchl = 1
    for iter in range(maxiter):
        if foqopt == "QMOREL":
            foq0 = foqint_morel(fqfile=fqfile, wave=bands, sza=solz, vzap=np.zeros_like(solz), reaa=np.zeros_like(solz),
                                chl=chl)
        else:
            foq0 = foqint_morel(fqfile=fqfile, wave=bands, sza=np.zeros_like(solz), vzap=np.zeros_like(solz),
                                reaa=np.zeros_like(solz), chl=chl)
        foq = foqint_morel(fqfile=fqfile, wave=bands, sza=solz, vzap=senzp, reaa=phip, chl=chl)

        brdf=foq0/foq
        brdf[chl < 0]=1.
        rrs=nlw*brdf/F0

        chl = get_chl.get_default_chl(rrs=rrs, b443=b443, b490=b490, b520=b520, b555=b555, b670=b670)
    brdf[np.isnan(mask)]=1.
    brdf[np.isnan(mask2)]=np.nan

    return brdf
"""