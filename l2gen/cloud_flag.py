# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/16 14:42
@FileName: cloud_flag.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
"""
char get_cloudmask_meris(l1str *l1rec, int32_t ip) {
    // Cloud Masking for MERIS

    static int ib443, ib490, ib620, ib665, ib681, ib709, ib754, ib865, ib885, firstCall = 1;
    int ipb;
    float *rhos = l1rec->rhos, *cloud_albedo = l1rec->cloud_albedo;
    float ftemp, cldtmp;
    char flagcld;

    if (firstCall == 1) {
        ib443 = bindex_get(443);
        ib490 = bindex_get(490);
        ib620 = bindex_get(620);
        ib665 = bindex_get(665);
        ib681 = bindex_get(681);
        ib709 = bindex_get(709);
        ib754 = bindex_get(754);
        ib865 = bindex_get(865);
        ib885 = bindex_get(885);

        if (ib443 < 0 || ib490 < 0 || ib620 < 0 || ib665 < 0 || ib681 < 0 || ib709 < 0 || ib754 < 0 || ib865 < 0 || ib885 < 0) {
            printf("get_habs_cldmask: incompatible sensor wavelengths for this algorithm\n");
            exit(EXIT_FAILURE);
        }
        firstCall = 0;
    }
    flagcld = 0;
    ipb = l1rec->l1file->nbands*ip;

    if (rhos[ipb + ib443] >= 0.0 &&
        rhos[ipb + ib620] >= 0.0 &&
        rhos[ipb + ib665] >= 0.0 &&
        rhos[ipb + ib681] >= 0.0 &&
        rhos[ipb + ib709] >= 0.0 &&
        rhos[ipb + ib754] >= 0.0) {
        // turbidity signal in water
        ftemp = (rhos[ipb + ib620] + rhos[ipb + ib665] + rhos[ipb + ib681]) - 3 * rhos[ipb + ib443] - \
                (rhos[ipb + ib754] - rhos[ipb + ib443]) / (754 - 443)*(620 + 665 + 681 - 3 * 443);
//     cldtmp = cloud_albedo[ip] - 3 * ftemp;
        //switch to rhos_865 where cldalb fails
        if (cloud_albedo[ip] >= 0.0) {
            cldtmp = cloud_albedo[ip];
        } else {
            cldtmp = rhos[ipb + ib865];
        }

        //remove turbidity signal from cloud albedo
        if (ftemp > 0) cldtmp = cldtmp - 3 * ftemp;

        if (cldtmp > 0.08) {
            flagcld = 1;
        }

        // to deal with scum look at relative of NIR and blue for lower albedos
        if ((rhos[ipb + ib754] + rhos[ipb + ib709]) > (rhos[ipb + ib443] + rhos[ipb + ib490]) && cldtmp < 0.1) flagcld = 0;
        if (((rhos[ipb + ib754] + rhos[ipb + ib709]) - (rhos[ipb + ib665] + rhos[ipb + ib681])) > 0.01 && cldtmp < 0.15) flagcld = 0;
        if ((((rhos[ipb + ib754] + rhos[ipb + ib709]) - (rhos[ipb + ib665] + rhos[ipb + ib681])) / cldtmp) > 0.1) flagcld = 0;
        if ((rhos[ipb + ib665] > 0.1) && (cldtmp > 0.15)) {
            flagcld = 1;
        }
    }
    return (flagcld);
}
"""