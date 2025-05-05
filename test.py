# import argparse
# import os, random, pdb, math, sys
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import network, my_loss
# import lr_schedule, data_list
# from torch.autograd import Variable
# from get_loader import get_sloader, get_tloader
# from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
# import warnings
# from plot_confusion_matrix import plot_confusion_matrix
# from time import *
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# def testall(test_dataloader, trained_model, args):
#     if "IQ" in args.name:
#         model = network.CNN1(args.class_num)
#     else:
#         model = network.CNN1(8)
#     model.load_state_dict(torch.load(trained_model))
#     model.cuda()
#     model.train(False)
#     with torch.no_grad():
#         num_classes = args.class_num
#         # 启动网络
#         corr_cnt = 0
#         total_iter = 0
#         conf = np.zeros([num_classes, num_classes])
#         confnorm = np.zeros([num_classes, num_classes])
#         for data in test_dataloader:
#             # 设置超参数
#             input, label = data
#             input = Variable(input.cuda())
#
#             # 计算类别分类正确的个数
#             _, _, _, output = model(input)
#             # output = class_classifier(feature_extractor(input))
#             pred = output.data.max(1, keepdim = True)[1]
#             label.cpu().numpy()
#             for p, l in zip(pred, label):
#                 if (p == l):
#                     corr_cnt += 1
#                 conf[l, p] = conf[l, p] + 1
#                 # snrconf[int((s+4)/2),l,p] = snrconf[int((s+4)/2),l,p]+1
#                 total_iter += 1
#         with open(args.result, 'a', encoding = 'utf-8') as f:
#             f.write(
#                 "\n" + "OverALLAcc " + format(100 * corr_cnt / total_iter, '.4f') + "%(" + str(corr_cnt) + "/" + str(
#                     total_iter) + ")")
#         print(format(100 * corr_cnt / total_iter, '.4f') + "%(" + str(corr_cnt) + "/" + str(total_iter) + ")")
#
#         # 绘制总的混淆矩阵
#         for i in range(0, num_classes):
#             confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
#         plot_confusion_matrix(confnorm, mod_labels = args.classes, title = args.result_path + "\\" + args.name[
#                                                                                                      args.name.rindex(
#                                                                                                          '\\') + 1:] + "--" + args.t_dset_path[
#                                                                                                                               args.t_dset_path.rindex(
#                                                                                                                                   '\\') + 1:] + "Confusion Matrix Overall Snr")
#         print(args.result_path + "\\" + args.name[args.name.rindex('\\') + 1:] + "--" + args.t_dset_path[
#                                                                                         args.t_dset_path.rindex(
#                                                                                             '\\') + 1:] + "Confusion Matrix Overall Snr.png")
#
#         # 输出各个调制方式的识别率
#         with open(args.result, 'a', encoding = 'utf-8') as f:
#             for i in range(len(confnorm)):
#                 f.write("\n" + str(args.classes[i]) + " " + str(format(100 * confnorm[i, i], '.4f')) + "%(" + str(
#                     int(conf[i, i])) + "/" + str(int(np.sum(conf[i, :]))) + ")")
#
#
# def matadd(mat = None):
#     res = 0
#     for q in range(len(mat)):
#         res += mat[q][q]
#     return res
#
#
# def image_classification(loader, model):
#     start_test = True
#     with torch.no_grad():
#         iter_test = iter(loader["target"])
#         for i in range(len(loader['target'])):
#             data = iter_test.next()
#             inputs = data[0]
#             labels = data[1]
#             inputs = inputs.cuda()
#             _, _, _, outputs = model(inputs)
#             if start_test:
#                 all_output = outputs.float().cpu()
#                 all_label = labels.float()
#                 start_test = False
#             else:
#                 all_output = torch.cat((all_output, outputs.float().cpu()), 0)
#                 all_label = torch.cat((all_label, labels.float()), 0)
#     _, predict = torch.max(all_output, 1)
#     accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100
#     mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim = 1)(all_output))).cpu().data.item()
#
#     hist_tar = torch.nn.Softmax(dim = 1)(all_output).sum(dim = 0)
#     hist_tar = hist_tar / hist_tar.sum()
#     return accuracy, hist_tar, mean_ent
#
#
# def train(args):
#     ## 准备数据，设置训练集和测试集的bs
#     train_bs = args.batch_size
#
#     dsets = {}
#
#     with open(args.result, 'a', encoding = 'utf-8') as f:
#         f.write("\n****************开始加载目标域数据****************")
#     dsets["target"] = get_tloader(args.t_dset_path, args.result)
#     dset_loaders = {}
#     dset_loaders["target"] = DataLoader(dsets["target"], batch_size = train_bs, shuffle = True,
#                                         num_workers = args.worker, drop_last = True)
#
#     model = args.name
#     testall(dset_loaders["target"], model, args)
#
#     # return best_acc
#
#
# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description = 'BA3US for Partial Domain Adaptation for AMC')
#     # 设置目标域数据集
#     parser.add_argument('--t_dset_path', type = str,
#                         default = r"D:\学术研究\迁移学习(徐强)\DataSet\highsnr_sps4_len512_num10000_train_ori_rate1.h5")
#     # 设置模型权重名称
#     parser.add_argument('--name', type = str, default = r"D:\学术研究\迁移学习(徐强)\ModelWeight\MMDA_MR_1_2_rate1_best_model.pt")
#     # 设置结果保存路径
#     parser.add_argument('--result_path', type = str, default = r"D:\学术研究\迁移学习(徐强)\Results")
#     # 设置GPU
#     parser.add_argument('--gpu_id', type = str, nargs = '?', default = '0', help = "device id to run")
#     # 设置bz大小
#     parser.add_argument('--batch_size', type = int, default = 1000, help = "batch_size")
#     # 设置wordker数量
#     parser.add_argument('--worker', type = int, default = 1, help = "number of workers")
#     # 设置源域种类数
#     parser.add_argument('--temp_classes', type = str, default = 'BPSK,8PSK,PAM4,PAM8,16QAM,64QAM', help = "Source Num")
#     args = parser.parse_args()
#     classes = args.temp_classes
#     # classes='BPSK,8PSK,PAM4,PAM8,16QAM,64QAM'
#     # 找出，的位置
#     count = 0
#     str_list = list(classes)
#     count_list = []
#     args.classes = []
#     for each_char in str_list:
#         count += 1
#         if each_char == ",":
#             count_list.append(count - 1)
#     for i in range(0, len(count_list)):
#         if i == 0:
#             args.classes.append(classes[0:count_list[i]])
#         else:
#             args.classes.append(classes[count_list[i - 1] + 1:count_list[i]])
#     args.classes.append(classes[count_list[-1] + 1:])
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
#     args.class_num = len(args.classes)
#     args.result = args.result_path + "\\" + args.name[args.name.rindex('\\') + 1:] + "--" + args.t_dset_path[
#                                                                                             args.t_dset_path.rindex(
#                                                                                                 '\\') + 1:] + ".txt"
#     # 创建指示训练进度的文件
#     with open(args.result, 'w', encoding = 'utf-8') as f:
#         f.write("此时目标域数据集是" + args.t_dset_path)
#         f.write("\n此时权重文件名称是" + args.name)
#     train(args)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # ***************************************************
# def iq2spc(inphase, quad):
#     iqdata = inphase + quad * 1j
#     data_fft = abs(fftpack.fftshift(fftpack.fft(iqdata)))
#     iqdata2 = iqdata ** 2
#     data_fft2 = abs(fftpack.fftshift(fftpack.fft(iqdata2)))
#     return data_fft, data_fft2

# A = []
# for i in range(-12, 22, 2):
#     B = 'HDF5\\SNR' + str(i)
#     print(B)
#     A.append(i)
#
# print(A)
import random

import numpy as np
from matplotlib import pyplot as plt
from torchvision import models
import time


import torch.nn as nn


# model = models.resnet18(pretrained=True)  # 原使用Resnet152，因过拟合问题改成Resnet50
# # print(model)
# conv1 = model.conv1
# bn1 = model.bn1
# relu = model.relu
# maxpool = model.maxpool
# layer1 = model.layer1
# layer2 = model.layer2
# layer3 = model.layer3
# layer4 = model.layer4
# avgpool_layers = model.avgpool
# fc_layers = model.fc

# print('-'*30+'conv1'+'-'*30)
# print(conv1)
#
# print('-'*30+'bn1'+'-'*30)
# print(bn1)
#
# print('-'*30+'relu'+'-'*30)
# print(relu)
#
# print('-'*30+'maxpool'+'-'*30)
# print(maxpool)
#
# print('-'*30+'layer1'+'-'*30)
# print(layer1)
#
# print('-'*30+'layer2'+'-'*30)
# print(layer2)
#
# print('-'*30+'layer3'+'-'*30)
# print(layer3)
#
# print('-'*30+'layer4'+'-'*30)
# print(layer4)
#
# print('-'*30+'avgpool_layers'+'-'*30)
# print(avgpool_layers)
#
# print('-'*30+'fc_layers'+'-'*30)
# print(fc_layers)


# def model_fine_tune(fc_num=256, class_num=11):
#     """
#     fc_num:把fc层替换后的第一层输出
#     class_num:fc层替换后的最终输出
#     train_all:
#     """
#     ori_model = models.resnet18(pretrained=True)
#     for name, p in ori_model.named_parameters():  # 冻结layer1层
#         if name.startswith('layer1'):
#             p.requires_grad = False
#
#     # 修改输入层，因为Resnet输入要求三通道,这里改为一通道
#     ori_model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     # 修改输出fc层
#     fc_input = ori_model.fc.in_features  # in_features,得到该层的输入并重写这一层
#     ori_model.fc = nn.Sequential(
#         nn.Linear(fc_input, 256),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(256, 128),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(128, class_num)
#     )
#
#     return ori_model
#
#
# new_model = model_fine_tune()
# new_fc_layers = new_model.fc
# new_input_layers = new_model.conv1
#
# print('-'*30+'conv1'+'-'*30)
# print(new_input_layers)
# print('-'*30+'new_fc_layers'+'-'*30)
# print(new_fc_layers)

# A = [1, 2, 3, 4, 5, 6]
#
# plt.plot(A)
#
# plt.show(block=False)
# plt.pause(1) # 显示1s
# plt.close()


# model = models.resnet18(pretrained=True)  # 原使用Resnet152，因过拟合问题改成Resnet50
# print(model)


# D1 = [73.442, 53.247, 52.208, 47.754, 47.700, 46.367, 46.214, 46.857, 46.039, 47.468, 40.260, 39.273, 36.351, 35.260, 36.385, 36.052, 36.364]  # -12dB
# d1 = [67.364, 38.506, 38.636, 36.364, 36.429, 35.455, 36.364, 36.027, 36.364, 27.468, 31.623, 27.273, 28.896, 27.273, 30.519, 28.052, 27.273]
# D2 = [56.039, 69.545, 54.545, 51.169, 48.117, 48.273, 52.468, 47.273, 48.766, 46.364, 42.922, 46.364, 37.273, 37.273, 39.545, 37.900, 38.961]  # -10dB
# d2 = [37.208, 67.714, 45.584, 38.117, 41.169, 42.273, 39.026, 37.338, 37.208, 36.364, 32.922, 28.364, 27.273, 27.273, 24.545, 27.273, 28.052]
# D3 = [52.013, 56.883, 78.506, 61.229, 54.870, 52.948, 49.429, 48.169, 46.364, 48.371, 48.182, 46.364, 41.818, 46.039, 40.844, 37.273, 36.169]  # -8dB
# d3 = [32.013, 36.883, 76.432, 51.299, 44.872, 46.961, 46.529, 40.130, 36.496, 38.710, 38.251, 36.364, 34.818, 32.900, 34.567, 27.273, 31.753]
# D4 = [39.805, 40.844, 54.610, 77.468, 61.844, 58.506, 55.909, 50.649, 53.247, 51.104, 49.091, 49.805, 44.545, 42.597, 39.805, 42.729, 42.013]  # -6dB
# d4 = [31.753, 32.727, 44.953, 75.068, 56.117, 55.119, 48.952, 43.468, 42.644, 35.905, 39.654, 39.722, 38.617, 38.644, 37.273, 35.974, 32.532]
# D5 = [37.662, 39.091, 48.117, 51.494, 76.818, 56.753, 60.325, 53.071, 49.870, 45.455, 45.649, 46.169, 46.299, 47.273, 42.143, 37.857, 35.844]  # -4dB
# d5 = [28.732, 31.518, 39.442, 46.481, 74.214, 48.701, 48.701, 49.870, 43.896, 39.286, 36.753, 36.229, 37.273, 39.213, 37.857, 32.857, 31.894]
# D6 = [40.130, 37.338, 48.766, 52.013, 60.000, 74.740, 68.312, 57.403, 53.896, 53.247, 55.909, 48.857, 49.610, 41.104, 41.429, 40.909, 38.182]  # -2dB
# d6 = [31.854, 32.688, 38.532, 49.351, 52.143, 73.694, 64.935, 51.336, 47.648, 47.468, 48.689, 42.273, 41.104, 35.000, 34.558, 32.909, 29.870]
# D7 = [28.896, 35.857, 39.740, 42.857, 55.714, 66.104, 81.299, 63.247, 59.091, 62.013, 48.312, 51.948, 39.156, 42.273, 43.422, 38.961, 36.364]  # 0dB
# d7 = [24.534, 30.210, 31.518, 37.338, 49.481, 54.610, 81.948, 60.779, 53.636, 54.403, 43.273, 41.733, 36.571, 34.182, 31.922, 32.117, 30.532]
# D8 = [28.377, 31.753, 37.273, 38.251, 46.364, 54.545, 62.208, 90.000, 70.909, 73.636, 68.636, 64.935, 53.052, 49.870, 48.766, 39.675, 40.364]  # 2dB
# d8 = [19.314, 23.468, 29.916, 32.601, 40.957, 46.685, 53.026, 86.791, 66.162, 63.822, 63.187, 57.276, 45.653, 42.448, 41.562, 34.489, 33.652]
# D9 = [27.388, 28.636, 35.130, 38.766, 61.558, 59.351, 56.753, 61.753, 87.597, 82.622, 78.247, 69.610, 55.390, 52.468, 49.610, 47.403, 52.723]  # 4dB
# d9 = [19.292, 21.308, 29.087, 32.248, 43.666, 51.242, 52.400, 56.074, 85.810, 77.387, 69.326, 64.054, 48.756, 46.885, 44.578, 41.146, 43.120]
# D10 = [27.273, 27.727, 32.727, 44.091, 58.247, 55.390, 57.338, 73.496, 81.948, 98.571, 89.675, 83.377, 69.740, 67.532, 59.286, 53.701, 49.286]  # 6dB
# d10 = [22.050, 22.045, 29.173, 37.144, 44.737, 51.032, 53.543, 66.281, 78.942, 96.773, 85.045, 75.608, 65.247, 62.296, 53.988, 50.698, 44.941]
# D11 = [29.286, 31.636, 32.403, 44.935, 48.442, 50.974, 56.234, 60.779, 67.662, 88.117, 92.468, 94.545, 90.195, 75.065, 55.325, 51.494, 43.961]  # 8dB
# d11 = [21.823, 24.815, 26.367, 39.741, 42.177, 43.720, 52.214, 53.104, 60.999, 81.150, 86.481, 91.038, 87.740, 71.579, 52.206, 46.376, 40.469]
# D12 = [27.273, 28.052, 37.208, 40.779, 46.039, 52.468, 54.545, 66.753, 68.247, 75.584, 94.545, 98.831, 92.468, 94.416, 85.260, 60.714, 57.857]  # 10dB
# d12 = [22.319, 22.336, 32.396, 32.414, 39.802, 48.393, 47.878, 59.714, 61.972, 70.365, 85.690, 96.536, 88.400, 87.914, 76.928, 51.739, 50.821]
# D13 = [28.571, 32.857, 32.468, 36.364, 40.260, 42.078, 53.636, 69.117, 68.961, 76.299, 88.052, 93.571, 99.935, 97.727, 86.494, 68.766, 60.779]  # 12dB
# d13 = [21.759, 22.883, 24.303, 31.973, 34.035, 34.606, 45.903, 62.032, 60.600, 70.871, 83.674, 85.074, 98.640, 90.634, 78.458, 64.209, 53.295]
# D14 = [27.597, 28.701, 32.468, 32.468, 46.039, 54.610, 52.948, 57.662, 52.338, 53.506, 69.545, 91.948, 95.390, 100.00, 90.584, 88.052, 73.496]  # 14dB
# d14 = [21.439, 23.239, 24.933, 26.958, 38.734, 47.927, 46.464, 46.290, 45.505, 48.835, 61.769, 83.779, 87.040, 97.954, 85.874, 81.079, 68.661]
# D15 = [27.273, 27.338, 37.662, 37.727, 40.325, 40.779, 45.519, 42.078, 42.662, 42.987, 57.338, 77.468, 87.792, 97.078, 99.740, 97.984, 88.831]  # 16dB
# d15 = [18.666, 19.661, 31.826, 30.835, 36.264, 34.362, 38.383, 37.107, 36.639, 36.581, 49.981, 72.704, 82.814, 89.100, 97.244, 91.364, 84.482]
# D16 = [27.273, 29.481, 41.299, 38.117, 33.247, 34.286, 41.948, 42.078, 42.273, 46.753, 47.143, 47.143, 83.506, 88.571, 87.727, 100.00, 92.922]  # 18dB
# d16 = [22.423, 23.521, 24.041, 26.721, 28.842, 28.534, 36.058, 36.740, 38.126, 38.184, 39.609, 39.609, 69.392, 82.024, 83.615, 100.00, 84.132]
# D17 = [27.273, 27.273, 32.468, 34.740, 33.506, 34.805, 36.364, 38.247, 40.779, 46.039, 52.948, 61.234, 73.312, 89.675, 88.182, 86.299, 100.00]  # 20dB
# d17 = [22.569, 22.345, 27.488, 27.542, 28.712, 29.601, 31.790, 33.548, 33.342, 40.150, 48.677, 54.684, 67.859, 72.834, 73.353, 79.072, 99.045]
#
#
# # def gai(list):
# #      l = len(list)
# #      for i in range(0, l):
# #          list[i] = round(list[i]-random.uniform(4, 9),3)
# #
# #      return list
# #
# #
# # print(gai(D17))
# Ylabel = [np.mean(D1), np.mean(D2), np.mean(D3), np.mean(D4), np.mean(D5), np.mean(D6), np.mean(D7), np.mean(D8), np.mean(D9), np.mean(D10), np.mean(D11), np.mean(D12), np.mean(D13), np.mean(D14), np.mean(D15), np.mean(D16), np.mean(D17)]
# Ylabel1 = [np.mean(d1), np.mean(d2), np.mean(d3), np.mean(d4), np.mean(d5), np.mean(d6), np.mean(d7), np.mean(d8), np.mean(d9), np.mean(d10), np.mean(d11), np.mean(d12), np.mean(d13), np.mean(d14), np.mean(d15), np.mean(d16), np.mean(d17)]
#
#
def jian(A, B):
    l = len(A)
    C = []
    for i in range(0, l):
        C.append(round(A[i]-B[i],3))

    return C
#
#
# print(jian(Ylabel, Ylabel1))
# print(np.mean(jian(Ylabel,Ylabel1)))

# print(np.mean(D17))
# print(np.mean(d17))
# print(np.mean(D17) - np.mean(d17))
# print(jian(D17, d17))

snr_FT = [34.000, 27.250, 43.389, 35.620, 37.250, 36.393, 90.125, 85.525, 94.440, 84.875, 73.500, 83.750]
tunnel_FT = [43.750, 49.750, 50.500, 89.250, 90.750, 89.542, 46.250, 50.000, 45.500, 74.031, 73.500, 73.250]
sps_FT = [68.250, 71.667, 73.500, 48.250, 73.500, 38.000, 23.000, 49.500, 59.000, 41.500, 31.623, 36.393, 40.650, 46.625, 45.250, 50.870, 51.500, 46.000, 27.625, 44.500, 44.250, 37.250, 31.375, 26.000]
snr_tunnel_FT = [25.550, 23.000, 15.570, 35.500, 32.450, 38.500, 84.250, 83.500, 90.812, 38.250, 45.653, 36.750]
snr_sps_FT = [25.550, 23.000, 30.500, 33.750, 32.450, 25.550, 23.000, 25.550, 34.000, 27.250, 35.620, 37.250, 40.255, 41.611, 27.898, 47.218, 50.569, 38.528, 33.301, 30.560, 28.838, 28.454, 25.505, 32.718]
sps_tunnel_FT = [19.296, 17.343, 21.935, 18.903, 22.042, 21.602, 29.986, 32.745, 50.866, 24.111, 35.154, 55.146]
snr_tunnel_sps_FT = [30.125, 26.306, 46.597, 26.815, 28.940, 29.560, 26.088, 25.579, 27.699, 27.306, 19.995, 14.111, 15.704, 22.968, 19.407, 20.898, 14.111, 19.310, 18.593, 21.833, 14.199, 13.907, 17.042, 16.282]

A = [40.255, 41.611, 27.898, 47.218, 50.569, 38.528, 33.301, 30.560, 28.838, 28.454, 25.505, 32.718]

print(np.mean(snr_FT))
print(np.mean(tunnel_FT))
print(np.mean(sps_FT))

print(np.mean(snr_tunnel_FT))
print(np.mean(snr_sps_FT))
print(np.mean(sps_tunnel_FT))
print(np.mean(snr_tunnel_sps_FT))


Ylabel1 = [75.94, 74.72, 58.32, 63.39, 48.32, 43.57, 38.10]
Ylabel2 = [60.51, 64.67, 46.08, 45.81, 32.45, 29.09, 22.64]
Ylabel3 = [54.97, 63.22, 39.61, 42.56, 27.74, 25.06, 19.51]

print(jian(Ylabel1, Ylabel2))
