from torchvision import models
import torch.nn as nn


model = models.resnet18(pretrained=True)  # 原使用Resnet152，因过拟合问题改成Resnet50
# print(model)
conv1 = model.conv1
bn1 = model.bn1
relu = model.relu
maxpool = model.maxpool
layer1 = model.layer1
layer2 = model.layer2
layer3 = model.layer3
layer4 = model.layer4
avgpool_layers = model.avgpool
fc_layers = model.fc

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


# Fine-Tune代码
def model_fine_tune(fc_num=256, class_num=11):
    """
    fc_num:把fc层替换后的第一层输出
    class_num:fc层替换后的最终输出
    train_all:
    """
    ori_model = models.resnet18(pretrained=True)
    for name, p in ori_model.named_parameters():
        if name.startswith('layer1'):
            p.requires_grad = False

    # 修改输入层，因为Resnet输入要求三通道,这里改为一通道
    ori_model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # 修改输出fc层
    fc_input = ori_model.fc.in_features  # in_features,得到该层的输入并重写这一层
    ori_model.fc = nn.Sequential(
        nn.Linear(fc_input, class_num),
        # nn.ReLU(),
        # # nn.Dropout(0.5),
        # nn.Linear(256, 128),
        # nn.ReLU(),
        # # nn.Dropout(0.5),
        # nn.Linear(128, class_num)
    )

    return ori_model


new_model = model_fine_tune()
new_fc_layers = new_model.fc
new_input_layers = new_model.conv1
# print('-'*30+'conv1'+'-'*30)
# print(new_input_layers)
# print('-'*30+'new_fc_layers'+'-'*30)
# print(new_fc_layers)