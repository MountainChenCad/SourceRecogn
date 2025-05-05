import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


# 域对抗损失
def DANN(features, ad_net, entropy=None, coeff=None, cls_weight=None, len_share=0):
    ad_out = ad_net(features)
    train_bs = (ad_out.size(0) - len_share) // 2
    # 设置域标签
    dc_target = torch.from_numpy(np.array([[1]] * train_bs + [[0]] * (train_bs + len_share))).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
    else:
        entropy = torch.ones(ad_out.size(0)).cuda()
    # 将目标域掩盖住，获取域对抗的源域样本权重,cls_weight是类级权重
    source_mask = torch.ones_like(entropy)
    source_mask[train_bs : 2 * train_bs] = 0
    source_weight = entropy * source_mask
    source_weight = source_weight * cls_weight
    # 将源域样本掩盖住，获取域对抗的目标域样本权重
    target_mask = torch.ones_like(entropy)
    target_mask[0 : train_bs] = 0
    target_mask[2 * train_bs::] = 0
    target_weight = entropy * target_mask
    target_weight = target_weight * cls_weight
    weight = (1.0 + len_share / train_bs) * source_weight / (torch.sum(source_weight).detach().item()) + \
            target_weight / torch.sum(target_weight).detach().item()
        
    weight = weight.view(-1, 1)
    # 获取域对抗的损失函数
    return torch.sum(weight * nn.BCELoss(reduction='none')(ad_out, dc_target)) / (1e-8 + torch.sum(weight).detach().item())


# 边缘损失
def marginloss(yHat, y, classes=65, alpha=1, weight=None):
    batch_size = len(y)
    classes = classes
    yHat = F.softmax(yHat, dim=1)
    # 从批量tensor中获取指定索引下的数据，熵期望标记源样本的不正确类别具有一致且低的预测分数。
    Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))#.detach()
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = yHat / Yg_.view(len(yHat), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    # 转换成独热码，有数值的为0
    y_zerohot = torch.ones(batch_size, classes).scatter_(
        1, y.view(batch_size, 1).data.cpu(), 0)

    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output, dim=1)/ np.log(classes - 1)
    Yg_ = Yg_ ** alpha
    if weight is not None:
        weight *= (Yg_.view(len(yHat), )/ Yg_.sum())
    else:
        weight = (Yg_.view(len(yHat), )/ Yg_.sum())

    weight = weight.detach()
    loss = torch.sum(weight * loss) / torch.sum(weight)

    return loss