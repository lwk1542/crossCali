import numpy as np
from scipy import interpolate

def interp(lat_ref, lon_ref, lat, lon, interp_keys: list, dataset: dict, method: str = "linear"):
    lat_ref = np.asarray(lat_ref, dtype=float)
    lon_ref = np.asarray(lon_ref, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat_ref.shape != lon_ref.shape:
        raise ValueError(f"lat_ref 和 lon_ref shape 不一致: {lat_ref.shape}, {lon_ref.shape}")

    if lat.shape != lon.shape:
        raise ValueError(f"lat 和 lon shape 不一致: {lat.shape}, {lon.shape}")

    # 参考点有效性
    valid_ref_geo = (
        np.isfinite(lat_ref) &
        np.isfinite(lon_ref) &
        (lat_ref >= -90) & (lat_ref <= 90)
    )

    # 目标点有效性
    valid_tar_geo = (
        np.isfinite(lat) &
        np.isfinite(lon) &
        (lat >= -90) & (lat <= 90)
    )

    if not np.any(valid_ref_geo):
        raise ValueError("lat_ref/lon_ref 中没有有效经纬度点")

    if not np.any(valid_tar_geo):
        raise ValueError("lat/lon 中没有有效经纬度点")

    points_tar = (
        lat[valid_tar_geo],
        lon[valid_tar_geo]
    )

    interp_results = {}

    for key in interp_keys:
        data = np.asarray(dataset[key])

        # -----------------------------
        # 情况 1：二维数据，shape = (row, col)
        # -----------------------------
        if data.ndim == 2 and data.shape == lat_ref.shape:
            data_float = data.astype(float)

            valid = valid_ref_geo & np.isfinite(data_float)

            points_ref = (
                lat_ref[valid],
                lon_ref[valid]
            )

            values_ref = data_float[valid]

            out = np.full(lat.shape, np.nan, dtype=float)

            interp_value = interpolate.griddata(
                points_ref,
                values_ref,
                points_tar,
                method=method
            )

            out[valid_tar_geo] = interp_value
            interp_results[key] = out

        # -----------------------------
        # 情况 2：三维数据，shape = (row, col, band)
        # -----------------------------
        elif data.ndim == 3 and data.shape[0:2] == lat_ref.shape:
            nband = data.shape[2]
            out = np.full(lat.shape + (nband,), np.nan, dtype=float)

            for i in range(nband):
                band_data = data[:, :, i].astype(float)

                valid = valid_ref_geo & np.isfinite(band_data)

                if not np.any(valid):
                    continue

                points_ref = (
                    lat_ref[valid],
                    lon_ref[valid]
                )

                values_ref = band_data[valid]

                interp_value = interpolate.griddata(
                    points_ref,
                    values_ref,
                    points_tar,
                    method=method
                )

                out[:, :, i][valid_tar_geo] = interp_value

            interp_results[key] = out

        # -----------------------------
        # 情况 3：三维数据，shape = (band, row, col)
        # -----------------------------
        elif data.ndim == 3 and data.shape[1:3] == lat_ref.shape:
            nband = data.shape[0]
            out = np.full((nband,) + lat.shape, np.nan, dtype=float)

            for i in range(nband):
                band_data = data[i, :, :].astype(float)

                valid = valid_ref_geo & np.isfinite(band_data)

                if not np.any(valid):
                    continue

                points_ref = (
                    lat_ref[valid],
                    lon_ref[valid]
                )

                values_ref = band_data[valid]

                interp_value = interpolate.griddata(
                    points_ref,
                    values_ref,
                    points_tar,
                    method=method
                )

                out[i, :, :][valid_tar_geo] = interp_value

            interp_results[key] = out

        else:
            raise ValueError(
                f"{key} 的维度无法和 lat_ref/lon_ref 匹配: "
                f"data.shape={data.shape}, lat_ref.shape={lat_ref.shape}"
            )

    return interp_results