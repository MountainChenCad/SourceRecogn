from torchvision import models
import torchvision
import torch.nn as nn
import torch


import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import network, my_loss
import lr_schedule, data_list
from torch.autograd import Variable
from get_loader import get_sloader, get_tloader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
import warnings
from plot_confusion_matrix import plot_confusion_matrix
from time import *


#
# print("Model's state_dict:")
# print(model)


# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

new_model = models.resnet152(pretrained=True)
# print(new_model)

print('-' * 30 + 'VGG卷积网络' + '-' * 30)
vgg16_ture = torchvision.models.vgg16(pretrained = True)
print(vgg16_ture)





net = models.vgg16()


class VGGnet(nn.Module):
    def __init__(self, feature_extract=True, num_classes=5):
        super(VGGnet, self).__init__()
        # 导入VGG16模型
        model = models.vgg16(pretrained=True)
        # 加载features部分
        self.features = model.features
        # 固定特征提取层参数
        set_parameter_requires_grad(self.features, feature_extract)
        # 加载avgpool层
        self.avgpool = model.avgpool
        # 改变classifier：分类输出层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        out = self.classifier(x)
        return out


# 固定参数，不进行训练
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


print('-' * 30 + 'CNN网络' + '-' * 30)
model = network.CNN()
model1 = network.CNN1()
model2 = network.AdversarialNetwork(128, 128)
print(model)
print('-' * 30 + 'CNN1网络' + '-' * 30)
print(model1)
print('-' * 30 + 'CNN2网络' + '-' * 30)
print(model2)








# trash

# def train(args):
#     # 准备数据，设置训练集和测试集的bs
#     train_bs = args.batch_size
#
#     dsets = {}
#     with open(args.trainlog, 'a', encoding='utf-8') as f:
#         f.write("\n****************开始加载源域数据****************")
#     dsets["source"] = get_sloader(args.s_dset_path, args.trainlog)
#     with open(args.trainlog, 'a', encoding='utf-8') as f:
#         f.write("\n****************开始加载目标域数据****************")
#     dsets["target"] = get_tloader(args.t_dset_path, args.trainlog)
#
#     dset_loaders = {}
#     dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,
#                                         num_workers=args.worker, drop_last=True)
#     dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
#                                         num_workers=args.worker, drop_last=True)
#     # 网络选择为更改后的Resnet网络
#     network = model_fine_tune()
#     # 优化器
#     optimizer = torch.optim.Adam(list(network.parameters()), lr=args.lr)
#     # 设置学习率调整方式
#     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
#     # 类级别权重
#     class_weight = None
#     best_ent = 1000
#     best_acc = 0
#     total_epochs = args.max_iterations // args.test_interval
#     with open(args.trainlog, 'a', encoding='utf-8') as f:
#         f.write("\n****************开始训练****************")
#     begin_time = time()
#     # 开始训练
#     for i in range(args.max_iterations + 1):
#         network.train(True)
#         # 更新学习率
#         scheduler.step()
#         if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
#             # test_interval 每进行n次迭代进行一次测试 相对的是caffe在训练过程中边训练边测试
#             # 获取当前的类级权重并且评估当前模型obtain the class-level weight and evalute the current model
#             network.train(False)
#             # 当前准确度，类别权重，平均熵
#             temp_acc, class_weight, mean_ent = image_classification(dset_loaders, network)
#             class_weight = class_weight.cuda().detach()
#
#             temp = [round(i, 4) for i in class_weight.cpu().numpy().tolist()]
#             log_str = str(temp)
#
#             if best_acc < temp_acc:
#                 best_ent, best_acc = mean_ent, temp_acc
#                 best_model = network.state_dict()
#                 log_str = "iter: {:05d}, temp best precision: {:.5f}, mean_entropy: {:.5f}".format(i, temp_acc,
#                                                                                                    mean_ent)
#
#                 with open(args.trainlog, 'a', encoding='utf-8') as f:
#                     f.write("\n" + log_str)
#                     f.write("\nSaving Model Weight...")
#                 torch.save(best_model, args.weigthpath + args.name + ".pt")
#             else:
#                 log_str = "iter: {:05d}, precision: {:.5f}, mean_entropy: {:.5f}".format(i, temp_acc, mean_ent)
#                 # args.out_file.write(log_str+"\n")
#                 # args.out_file.flush()
#                 with open(args.trainlog, 'a', encoding='utf-8') as f:
#                     f.write("\n" + log_str)
#
#         if i % args.test_interval == 0:
#             if args.mu > 0:  # mu ？？
#                 epoch = i // args.test_interval
#                 # 共享的样本个数，随着epoch的增长而减小
#                 len_share = int(max(0, (train_bs // args.mu) * (1 - epoch / total_epochs)))
#             elif args.mu == 0:
#                 len_share = 0  # no augmentation
#             else:
#                 len_share = int(train_bs // abs(args.mu))  # fixed augmentation
#             log_str = "\n{}, iter: {:05d}, source/ target/ middle: {:02d} / {:02d} / {:02d}\n".format("training ", i,
#                                                                                                       train_bs,
#                                                                                                       train_bs,
#                                                                                                       len_share)
#
#             with open(args.trainlog, 'a', encoding='utf-8') as f:
#                 f.write("\n" + log_str)
#                 # 设置共享的样本，随机选取len_share个
#             dset_loaders["middle"] = None
#             if not len_share == 0:
#                 dset_loaders["middle"] = DataLoader(dsets["source"], batch_size=len_share, shuffle=True,
#                                                     num_workers=args.worker,
#                                                     drop_last=True)
#                 iter_middle = iter(dset_loaders["middle"])
#
#         if i % len(dset_loaders["source"]) == 0:
#             iter_source = iter(dset_loaders["source"])
#         if i % len(dset_loaders["target"]) == 0:
#             iter_target = iter(dset_loaders["target"])
#         if dset_loaders["middle"] is not None and i % len(dset_loaders["middle"]) == 0:
#             iter_middle = iter(dset_loaders["middle"])
#         # 获取源域数据和标签，目标域数据
#         inputs_source, labels_source = iter_source.next()
#         inputs_target, labels_target = iter_target.next()
#         inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
#
#         if class_weight is not None and args.weight_cls and class_weight[labels_source].sum() == 0:
#             continue
#
#         features_source1, features_source2, outputs_source = network(inputs_source)
#         features_target1, features_target2, outputs_target = network(inputs_target)