#-*- coding:utf-8 -*-
import h5py
from h5py import Dataset, Group, File

# f = h5py.File('E:\\academic\\Data\\MZK_Dataset\\Test_Dataset\\0dB_len8192_sigsys11.h5', "a")
f = h5py.File('E:\\academic\\Doctor\\XQ_Project\\12domain\\train\\HighSnr_sps4_Awgn_len128_num1200_train.h5', "a")


# 读取HDF5文件内容
def read_h5_file_key(file, datasets_name_list, groups_name_list, sub_group):
    """
    对.h5文件读取，并根据目录下的分类(dataset和group)分别进行处理，并创建两个字典，键值对结构为 文件名：文件内容，最后返回两个字典
    :param sub_group: 添加包含子文件group的name的列表
    :param file:需要进行切割处理的.h5文件
    :param datasets_name_list:存储dataset数值和属性的字典
    :param groups_name_list:存储group的字典
    :return:
    之所以输入输出都有list是为了让在自循环过程中不丢失数值
    """
    # sub_group = []  # 该列表保存子文件下仍有dataset或group的group，并在之后return该程序进行二次检测
    for k in file.keys():
        if isinstance(file[k], Dataset):
            # print(file[k].name)
            datasets_name_list[file[k].name] = file[k][:]
        else:
            # print(f[k].name)
            groups_name_list[file[k].name] = file[k]
            print(groups_name_list)
            # print(str(file[k]))
            result = '0 members' in str(file[k])
            # print(result)
            if not result:
                print('Still have more group or dataset in ' + file[k].name + ' can be searched')
                sub_group.append(file[k].name)
                # print(file[k].name)
                # return read_h5_file_key(file[file[k].name], datasets_name_list, groups_name_list)  # 报错
            else:
                print('No more group or dataset can be searched')
    # print(sub_group)
    # print(file[sub_group[0]])
    if sub_group:
        for i in range(0, len(sub_group)):
            sub_datasets_name_list, sub_groups_name_list = read_h5_file_key(file[sub_group[i]], datasets_name_list,
                                                                    groups_name_list, sub_group)  # sub_group[i]是str格式
            datasets_name_list.update(sub_datasets_name_list)
            groups_name_list.updata(sub_groups_name_list)
    else:
        return datasets_name_list, groups_name_list


# read_h5_file_key()模块测试部分
A, B = read_h5_file_key(f, {}, {}, [])
# C = A['/Data_IQ'].T
keys = list(A.keys())
print(keys)
print('-------------------------')
# print(A, B)

