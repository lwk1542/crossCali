
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/23 17:59
@FileName: test.py
@Project : atmospheric_correction
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""
import numpy as np

# 创建一个示例的 4 维 NumPy 数组
array_4d = np.random.randint(0, 10, size=(2, 3, 4, 5))  # 一个随机的 4 维数组

# 创建一个示例的 4 维索引矩阵
index_matrix_dim1 = np.array([[0, 1, 0], [1, 1, 0]])[:, :, np.newaxis, np.newaxis]  # 第一个维度的索引
index_matrix_dim2 = np.array([[2, 1, 0], [1, 2, 0]])[:, :, np.newaxis, np.newaxis]  # 第二个维度的索引
index_matrix_dim3 = np.array([[3, 2, 1], [0, 2, 3]])[:, :, np.newaxis, np.newaxis]  # 第三个维度的索引
index_matrix_dim4 = np.array([[4, 0, 3], [1, 3, 2]])[:, :, np.newaxis, np.newaxis]  # 第四个维度的索引

# 使用 np.take 获取值
selected_values = np.take(array_4d, np.ravel_multi_index((index_matrix_dim1.flatten(),
                                                          index_matrix_dim2.flatten(),
                                                          index_matrix_dim3.flatten(),
                                                          index_matrix_dim4.flatten()),
                                                         dims=array_4d.shape))

# 将获取的值重新reshape成与索引矩阵相同的形状
selected_values = selected_values.reshape(index_matrix_dim1.shape)

# 打印获取的值
print(selected_values)


# 初始化一个空数组用于存储选中的值
selected_values2 = np.empty(index_matrix_dim1.shape, dtype=array_4d.dtype)

# 使用循环逐个索引获取值
for i in range(index_matrix_dim1.shape[0]):
    for j in range(index_matrix_dim1.shape[1]):
        indices = (index_matrix_dim1[i, j, 0, 0], index_matrix_dim2[i, j, 0, 0], index_matrix_dim3[i, j, 0, 0], index_matrix_dim4[i, j, 0, 0])
        selected_values2[i, j] = array_4d[indices]
print(selected_values2)



import timeit

# 要测试的代码片段
code_to_test = """
# 创建一个示例的 4 维 NumPy 数组
import numpy as np
array_4d = np.random.randint(0, 10, size=(2, 3, 4, 5))  # 一个随机的 4 维数组

# 创建一个示例的 4 维索引矩阵
index_matrix_dim1 = np.array([[0, 1, 0], [1, 1, 0]])[:, :, np.newaxis, np.newaxis]  # 第一个维度的索引
index_matrix_dim2 = np.array([[2, 1, 0], [1, 2, 0]])[:, :, np.newaxis, np.newaxis]  # 第二个维度的索引
index_matrix_dim3 = np.array([[3, 2, 1], [0, 2, 3]])[:, :, np.newaxis, np.newaxis]  # 第三个维度的索引
index_matrix_dim4 = np.array([[4, 0, 3], [1, 3, 2]])[:, :, np.newaxis, np.newaxis]  # 第四个维度的索引

# 使用 np.take 获取值
selected_values = np.take(array_4d, np.ravel_multi_index((index_matrix_dim1.flatten(),
                                                          index_matrix_dim2.flatten(),
                                                          index_matrix_dim3.flatten(),
                                                          index_matrix_dim4.flatten()),
                                                         dims=array_4d.shape))

# 将获取的值重新reshape成与索引矩阵相同的形状
selected_values = selected_values.reshape(index_matrix_dim1.shape)
print(selected_values)
# 初始化一个空数组用于存储选中的值
selected_values2 = np.empty(index_matrix_dim1.shape, dtype=array_4d.dtype)

# 使用循环逐个索引获取值
for i in range(index_matrix_dim1.shape[0]):
    for j in range(index_matrix_dim1.shape[1]):
        indices = (index_matrix_dim1[i, j, 0, 0], index_matrix_dim2[i, j, 0, 0], index_matrix_dim3[i, j, 0, 0], index_matrix_dim4[i, j, 0, 0])
        selected_values2[i, j] = array_4d[indices]
print(selected_values2)
# """
#
# # 运行代码片段并测量时间
# execution_time = timeit.timeit(code_to_test, number=1)  # number 表示执行次数
#
# # 打印代码执行时间
# print(f"Execution time: {execution_time} seconds")