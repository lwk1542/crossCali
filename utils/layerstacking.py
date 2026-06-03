# -*- coding: utf-8 -*-
"""
@Time    : 2025/1/7 21:10
@FileName: layerstacking.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
from osgeo import gdal
import os


def stack_layers(input_files_bands, reference_file, output_file, data_type=gdal.GDT_Float32, resample_alg=gdal.GRA_Bilinear):
    """
    将多个栅格文件的指定波段堆叠成一个多波段输出文件，按照参考文件的分辨率和空间参考。

    参数：
    - input_files_bands: dict
        输入文件及其对应要提取的波段。例如：
        {
            "input1.tif": [1, 3],
            "input2.tif": [2, 4],
            "input3.tif": [1, 2]
        }
    - reference_file: str
        参考文件的路径，用于统一分辨率和空间参考。
    - output_file: str
        输出堆叠后的文件路径。
    - data_type: GDAL 数据类型, 默认 gdal.GDT_Float32
        输出临时文件的数据类型。
    - resample_alg: GDAL 重采样算法, 默认 gdal.GRA_Bilinear
        重采样方法。
    """

    # 检查参考文件是否存在
    if reference_file not in input_files_bands:
        # 如果参考文件未在字典中，假设提取所有波段
        ref_ds = gdal.Open(reference_file)
        if ref_ds is None:
            raise FileNotFoundError(f"无法打开参考文件 {reference_file}")
        ref_bands = list(range(1, ref_ds.RasterCount + 1))
        input_files_bands[reference_file] = ref_bands
        ref_ds = None  # 关闭数据集

    # 获取参考文件的分辨率和地理信息
    ref_dataset = gdal.Open(reference_file)
    if ref_dataset is None:
        raise FileNotFoundError(f"无法打开参考文件 {reference_file}")

    ref_geo_transform = ref_dataset.GetGeoTransform()
    ref_projection = ref_dataset.GetProjection()
    ref_x_res = ref_geo_transform[1]
    ref_y_res = abs(ref_geo_transform[5])  # 通常是负值，取绝对值

    print(f"参考文件分辨率: x_res={ref_x_res}, y_res={ref_y_res}")

    # 创建一个列表来存储所有单波段的临时文件路径
    temp_single_band_files = []

    for idx, (file, bands) in enumerate(input_files_bands.items()):
        print(f"处理文件 {file} ({idx+1}/{len(input_files_bands)})")
        dataset = gdal.Open(file)
        if dataset is None:
            print(f"无法打开文件 {file}，跳过。")
            continue

        # 如果不是参考文件，则进行重采样
        if file != reference_file:
            resampled_file = f"resampled_{os.path.basename(file)}"
            print(f"重采样文件 {file} 到 {resampled_file}，分辨率: x_res={ref_x_res}, y_res={ref_y_res}")
            gdal.Warp(
                resampled_file,
                dataset,
                format='GTiff',
                xRes=ref_x_res,
                yRes=ref_y_res,
                targetAlignedPixels=True,
                resampleAlg=resample_alg,
                dstSRS=ref_projection  # 统一投影
            )
        else:
            resampled_file = file  # 参考文件无需重采样

        # 打开（重采样后的）文件
        resampled_dataset = gdal.Open(resampled_file)
        if resampled_dataset is None:
            print(f"无法打开重采样后的文件 {resampled_file}，跳过。")
            continue

        # 提取指定波段
        for band_num in bands:
            temp_file = f"temp_{os.path.basename(file).replace('.tif', '')}_band{band_num}.tif"
            gdal.Translate(temp_file, resampled_file, bandList=[band_num])
            # 保存单波段临时文件路径
            temp_single_band_files.append(temp_file)

        # 如果是重采样产生的临时文件（非参考文件），可以删除
        if file != reference_file:
            resampled_dataset = None
            os.remove(resampled_file)

    # 合并所有单波段的临时文件
    if temp_single_band_files:
        print(f"构建 VRT 文件用于合并波段")
        vrt_options = gdal.BuildVRTOptions(separate=True)
        vrt = gdal.BuildVRT("temp_stack.vrt", temp_single_band_files, options=vrt_options)

        if vrt is None:
            raise RuntimeError("无法创建 VRT 文件。请检查输入文件和波段。")

        print(f"将 VRT 转换为实际的 GeoTIFF 文件 {output_file}")
        gdal.Translate(output_file, vrt, format='GTiff')

        # 删除临时 VRT 文件
        vrt = None  # 关闭 VRT
        os.remove("temp_stack.vrt")

        print(f"输出已保存到 {output_file}")
    else:
        print("没有临时文件需要合并。")

    # 清理所有单波段临时文件
    for temp_file in temp_single_band_files:
        try:
            os.remove(temp_file)
        except OSError as e:
            print(f"无法删除临时文件 {temp_file}: {e}")

    print("所有临时文件已删除。")


# 示例用法
if __name__ == "__main__":
    import numpy as np

    # 定义输入影像文件及其对应要提取的波段
    input_files_bands = {
        "input1.tif": [1, 3],  # 从 input1.tif 提取波段1和波段3
        "input2.tif": [2, 4],  # 从 input2.tif 提取波段2和波段4
        "input3.tif": [1, 2]   # 从 input3.tif 提取波段1和波段2
    }

    # 指定参考文件（用于统一分辨率和空间参考）
    reference_file = "reference.tif"  # 请将此替换为您希望作为参考的文件名

    # 输出文件
    output_file = "output_stack.tif"

    # 调用函数进行堆叠
    stack_layers(input_files_bands, reference_file, output_file)
