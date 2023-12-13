# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/4 11:11
@FileName: get_filelist.py
@Project : git_repository
@Author  : 李文凯 liwenkai
@Email   : liwenkai@scsio.ac.cn/lwk1542@hotmail.com
@phone   : 132-9663-2830
"""


def get_filelist(idx: list[str], path: str = None, mode='all'):
    """
    输入父目录和至少一个能够指示文件名的字符串，比如‘.hdf’等
    Args:
        path (): 文件检索的父目录
        mode (): 关键字检索模式，或（any）/且（all），any表示任何一个关键字符匹配就可以
        idx (): 指示文件字符串列表，
    Returns:
        检索到的文件列表
    """
    import os
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 如果是需要包含任何一个字符串，则用any
            # 这里是不区分大小写的
            if mode == 'all':
                if all(idxi.lower() in filename.lower() for idxi in idx):
                    Filelist.append(os.path.join(home, filename))
            elif mode == 'any':
                if all(idxi.lower() in filename.lower() for idxi in idx):
                    Filelist.append(os.path.join(home, filename))
            else:
                print('the mode parameter was wrong and should be "all","any",or nothing')
    return Filelist

# def get_filelist(idx, *args, path=None, mode='all'):
#     """
#     输入父目录和至少一个能够指示文件名的字符串，比如‘.hdf’等
#     Args:
#         path (): 文件检索的父目录
#         mode (): 关键字检索模式，或（any）/且（all），any表示任何一个关键字符匹配就可以
#         idx (): 一个指示文件字符串，
#         *args (): 任意多个指示文件的字符串
#
#     Returns:
#         检索到的文件列表
#     """
#     import os
#     Filelist = []
#     for home, dirs, files in os.walk(path):
#         for filename in files:
#             # 如果是需要包含任何一个字符串，则用any
#             # 这里是不区分大小写的
#             if mode == 'all':
#                 if all(idxi.lower() in filename.lower() for idxi in args + (idx,)):
#                     Filelist.append(os.path.join(home, filename))
#             elif mode == 'any':
#                 if all(idxi.lower() in filename.lower() for idxi in args + (idx,)):
#                     Filelist.append(os.path.join(home, filename))
#             else:
#                 print('the mode parameter was wrong and should be "all","any",or nothing')
#     return Filelist


if __name__=='__main__':
    path = r'F:\DATA\SCS\manual_15_20_115_120/'
    files = get_filelist(path, 'MOD.1KM.TOA.', '.hdf')