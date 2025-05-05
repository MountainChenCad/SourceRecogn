import numpy as np
import torch
import torch.nn as nn


# from torchvision import models


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    """
    初始化权重
    Conv2d,ConvTranspose2d:二维卷积层
    BatchNorm:加入可训练的参数的归一化层
    Linear:线性层
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class CNN(nn.Module):
    """
    特征提取器: 主要包含5个卷积层（conv1，conv2，conv3，conv4，conv5）和1个全连接层（fc1）用于提取源域和目标域的特征，其中所有层均使用ReLU作为
    激活函数，且每一层卷积层后均使用BatchNorm进行归一化处理，从而加速网络收敛速度，同时在conv2，conv3，conv4，conv5这四个卷积层后加上池化层以减小
    数据的维度，从而简化网络复杂度、减小计算量。
--------------------------------------------------------------------------------------------------------------------------
    nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
    卷积层
    参数：
    in_channel:　输入数据的通道数，例RGB图片通道数为3；
    out_channel: 输出数据的通道数，这个根据模型调整；
    kennel_size: 卷积核大小，可以是int，或tuple；kennel_size=2,意味着卷积大小(2,2)， kennel_size=（2,3），意味着卷积大小（2，3）即非正方形卷积
    stride：步长，默认为1，与kennel_size类似，stride=2,意味着步长上下左右扫描皆为2， stride=（2,3），左右扫描步长为2，上下为3；
    padding：设置在所有边界增加 值为 0 的边距的大小（也就是在feature map 外围增加几圈 0 ），例如当 padding =1 的时候，如果原来大小为 3 × 3 ，那么之后的大小为 5 × 5 。即在外围加了一圈 0
    dilation：控制卷积核之间的间距
    groups：控制输入和输出之间的连接
    bias： 是否将一个 学习到的 bias 增加输出中，默认是 True
    padding_mode ： 字符串类型，接收的字符串只有 “zeros” 和 “circular”
    假设输入形状为(N,C_in,H_in,W_in),输出为(N,C_out,H_out,W_out),其中
    H_out = ((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1)-1)/stride[0]) + 1  向下取整
    W_out = ((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1)-1)/stride[1]) + 1  向下取整
-------------------------------------------------------------------------------------------------------------------------
    nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
    归一化函数
    参数:
    num_features:指特征数。 一般情况下输入的数据格式为（batch_size ，num_features ， height ， width）其中的C为特征数，也称channel数
    eps:为分数值稳定而添加到分母的值。 默认值：1e-5
    momentum:一个用于运行过程中均值和方差的一个估计参数。 可以将累积移动平均线（即简单平均线）设置为 None 。 默认值：0.1
    affine:一个布尔值，当设置为True时，此模块具有可学习的仿射参数。γ(gamma) 和 β(beta) （可学习的仿射变换参数） 默认值：True
    track_running_stats:一个布尔值，当设置为True时，此模块跟踪运行平均值和方差；设置为False时，此模块不跟踪此类统计信息，并将统计信息缓冲区
    running_mean和running_var初始化为None。 当这些缓冲区为None时，此模块将始终使用批处理统计信息。 在训练和评估模式下都可以。 默认值：True
-------------------------------------------------------------------------------------------------------------------------
    nn.ReLU(inplace=True)
    线性层（激活函数为ReLU）
    参数:
    inplace=True：inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出，默认为True
-------------------------------------------------------------------------------------------------------------------------
    nn.MaxPool2d(kernel_size, stride, padding, dilation,return_indices, ceil_mode)
    池化层
    参数:
    kernel_size: 表示做最大池化的窗口大小，可以是单个值，也可以是tuple元组
    stride: 步长，可以是单个值，也可以是tuple元组
    padding: 填充，可以是单个值，也可以是tuple元组
    dilation: 控制窗口中元素步幅
    return_indices: 布尔类型，返回最大值位置索引
    ceil_mode: 布尔类型，为True，用向上取整的方法，计算输出形状；默认是向下取整
-------------------------------------------------------------------------------------------------------------------------
    nn.Sequential()
    一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
-------------------------------------------------------------------------------------------------------------------------
    nn.AdaptiveAvgPool2d(int or tuple)
    自适应平均池化
-------------------------------------------------------------------------------------------------------------------------
    nn.Dropout(p)
    Dropout ，在训练时以一定的概率使神经元失活，实际上就是让对应神经元的输出为0,神经网络中的dropout可以作为正则化
    不能用于测试集，只能用于训练集
    参数:
    p: 神经元失活的概率

    """

    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 7), stride=2, padding=(0, 3), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 128)
        self.feature_layers = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, \
                                            self.conv5, self.avgpool)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout()
        )
        '''
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes))
            '''

        self.__in_features = 128

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class CNN1(nn.Module):
    """
    标签预测器：包含2层全连接层（fc1，fc2）用于预测源域数据的标签，所有层均使用ReLU作为激活函数，3个特征提取器输出的隐藏层特征在输入标签预测器前进行融合拼接
    """
    def __init__(self, num_classes=8):
        super(CNN1, self).__init__()
        self.Extractor1 = CNN(num_classes)  # 特征提取器
        self.Extractor2 = CNN(num_classes)
        self.Extractor3 = CNN(num_classes)    # 柯达  # MZK
        self.__in_features = 128
        self.classifier = nn.Sequential(
            nn.Linear(384, 192),    # 柯达  # MZK 256→384,128→192
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(192, 192),    # 柯达  # 128→192
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(192, num_classes))    # 柯达  # 128→192

    def forward(self, x):
        feature1 = self.Extractor1(x[:, :, 0:2])
        feature2 = self.Extractor2(x[:, :, 2:4])
        feature3 = self.Extractor3(x[:, :, 4:6])  # 柯达  # MZK
        feature = torch.cat([feature1, feature2, feature3], 1)    # 柯达
        output = self.classifier(feature)
        return feature1, feature2, feature3, output   # 柯达

    def output_num(self):
        return self.__in_features


class AdversarialNetwork(nn.Module):
    """
    域对抗器网络
    """
    def __init__(self, in_feature, hidden_size, max_iter=10000):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        # 对梯度进行反转
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)  # ?
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters()}]

# class ResNetFc(nn.Module):
#   def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
#     super(ResNetFc, self).__init__()
#     model_resnet = resnet_dict[resnet_name](pretrained=False)
#     self.conv1 = model_resnet.conv1
#     self.bn1 = model_resnet.bn1
#     self.relu = model_resnet.relu
#     self.maxpool = model_resnet.maxpool
#     self.layer1 = model_resnet.layer1
#     self.layer2 = model_resnet.layer2
#     self.layer3 = model_resnet.layer3
#     self.layer4 = model_resnet.layer4
#     self.avgpool = model_resnet.avgpool
#     self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
#                          self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
#
#     self.use_bottleneck = use_bottleneck
#     self.new_cls = new_cls
#     if new_cls:
#         if self.use_bottleneck:
#             self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
#             self.fc = nn.Linear(bottleneck_dim, class_num)
#             self.bottleneck.apply(init_weights)
#             self.fc.apply(init_weights)
#             self.__in_features = bottleneck_dim
#         else:
#             self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
#             self.fc.apply(init_weights)
#             self.__in_features = model_resnet.fc.in_features
#     else:
#         self.fc = model_resnet.fc
#         self.__in_features = model_resnet.fc.in_features
#
#   def forward(self, x):
#     x = self.feature_layers(x)
#     x = x.view(x.size(0), -1)
#     if self.use_bottleneck and self.new_cls:
#         x = self.bottleneck(x)
#     y = self.fc(x)
#     return x, y
#
#   def output_num(self):
#     return self.__in_features
#
#   def get_parameters(self):
#     if self.new_cls:
#         if self.use_bottleneck:
#             parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
#                             {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
#         else:
#             parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
#     else:
#         parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
#     return parameter_list

# vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn}
# class VGGFc(nn.Module):
#   def __init__(self, vgg_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
#     super(VGGFc, self).__init__()
#     model_vgg = vgg_dict[vgg_name](pretrained=True)
#     self.features = model_vgg.features
#     self.classifier = nn.Sequential()
#     for i in range(6):
#         self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
#     self.feature_layers = nn.Sequential(self.features, self.classifier)
#
#     self.use_bottleneck = use_bottleneck
#     self.new_cls = new_cls
#     if new_cls:
#         if self.use_bottleneck:
#             self.bottleneck = nn.Linear(4096, bottleneck_dim)
#             self.fc = nn.Linear(bottleneck_dim, class_num)
#             self.bottleneck.apply(init_weights)
#             self.fc.apply(init_weights)
#             self.__in_features = bottleneck_dim
#         else:
#             self.fc = nn.Linear(4096, class_num)
#             self.fc.apply(init_weights)
#             self.__in_features = 4096
#     else:
#         self.fc = model_vgg.classifier[6]
#         self.__in_features = 4096
#
#   def forward(self, x):
#     x = self.features(x)
#     x = x.view(x.size(0), -1)
#     x = self.classifier(x)
#     if self.use_bottleneck and self.new_cls:
#         x = self.bottleneck(x)
#     y = self.fc(x)
#     return x, y
#
#   def output_num(self):
#     return self.__in_features
#
#   def get_parameters(self):
#     if self.new_cls:
#         if self.use_bottleneck:
#             parameter_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
#                             {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
#         else:
#             parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
#     else:
#         parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
#     return parameter_list
