import argparse
import os, random, pdb, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import network, my_loss
import lr_schedule, data_list
from get_loader import get_sloader,get_tloader
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import warnings
from torch.autograd import Variable
import h5py
from plot_confusion_matrix import plot_confusion_matrix
warnings.filterwarnings("ignore")


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a",encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


type = sys.getfilesystemencoding()
# 最终测试结果


def testall(test_dataloader,trained_model,args):
  model=network.CNN1(args.class_num)
  model.load_state_dict(torch.load(trained_model))
  model.cuda()
  model.train(False)
  with torch.no_grad():
    num_classes = args.num_classes
    # 启动网络
    corr_cnt = 0
    total_iter = 0
    conf = np.zeros([num_classes, num_classes])
    confnorm = np.zeros([num_classes, num_classes])
    for data in test_dataloader:
        # 设置超参数
        input, label= data
        input = Variable(input.cuda())

        # 计算类别分类正确的个数
        _, _,_,output = model(input)
        #output = class_classifier(feature_extractor(input))
        pred = output.data.max(1, keepdim = True)[1]
        label.cpu().numpy()
        for p,l in zip(pred,label):
            if(p == l):
                corr_cnt += 1
            conf[l,p] = conf[l,p]+1
            #snrconf[int((s+4)/2),l,p] = snrconf[int((s+4)/2),l,p]+1
            total_iter +=1 
    print(args.name+" AccuracyOverall" + " "+ format(100*corr_cnt/total_iter,'.4f') + "%(" +str(corr_cnt) + "/"+ str(total_iter) + ")")
    #绘制总的混淆矩阵
    for i in range(0, num_classes):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plot_confusion_matrix(confnorm, mod_labels=args.classes,title=args.name+"Confusion Matrix Overall Snr")
    
    # 输出各个调制方式的识别率
    for i in range(len(confnorm)):
        print(str(args.classes[i])+" "+str(format(100*confnorm[i, i],'.4f'))+"%("+str(int(conf[i,i]))+"/"+str(int(np.sum(conf[i,:])))+")")


def matadd(mat=None):
  res = 0
  for q in range(len(mat)):
      res+=mat[q][q]
  return res


def image_train(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def image_test(resize_size=256, crop_size=224):
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["target"])
        for i in range(len(loader['target'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, _,_,outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])*100
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()
    return accuracy, hist_tar, mean_ent


def train(args):
    ## 准备数据，设置训练集和测试集的bs
    train_bs = args.batch_size

    dsets = {}
    dsets["source"]=get_sloader(args.s_dset_path)
    dsets["target"]=get_tloader(args.t_dset_path,args.rate)
    #dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())
    #dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())
    #dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())
    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    #定义特征提取器
    base_network=network.CNN1(args.class_num)
    base_network=base_network.cuda() 
    model=r"E:\xuqiang\ch5_final\Non-PartialV1\\"+args.name+"_best_model.pt"
    testall(dset_loaders["target"],model,args)

    #return best_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation for AMC')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # 设置源域
    parser.add_argument('--s', type=int, default=0, help="source")
    # 设置目标域
    parser.add_argument('--t', type=int, default=1, help="target")
    # 没什么用
    parser.add_argument('--output', type=str, default='run')
    # 设置随机种子
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    # 最大迭代次数
    parser.add_argument('--max_iterations', type=int, default=1000, help="max iterations")
    # 设置bz大小
    parser.add_argument('--batch_size', type=int, default=4000, help="batch_size")
    parser.add_argument('--worker', type=int, default=1, help="number of workers") 
    # 一个epoch包含的iter
    parser.add_argument('--test_interval', type=int, default=36, help="interval of two continuous test phase")
    # 学习率
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    # 初始的对齐大小，batch_size//mu
    parser.add_argument('--mu', type=int, default=0, help="init augmentation size = batch_size//mu")
    # 目标域条件熵系数
    parser.add_argument('--ent_weight', type=float, default=0)
    # 自适应不确定性抑制损失系数
    parser.add_argument('--cot_weight', type=float, default=0, choices=[0, 1, 5, 10])
    # 是否对adv使用类级别权重
    parser.add_argument('--weight_aug', type=bool, default=False)
    # 是否对cls使用类级别权重
    parser.add_argument('--weight_cls', type=bool, default=False)
    # ？
    parser.add_argument('--alpha', type=float, default=1)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.classes=["BPSK","8PSK","16QAM","64QAM","PAM4","PAM8"]
    args.num_classes = len(args.classes)
    __console__ = sys.stdout
    list_ori_source=[0]
    
    # list_ori_target=np.linspace(1,2,2,dtype=np.int8)
    rate_list=np.linspace(0,6,5,dtype=np.int8)
    # np.linspace(0,2,3,dtype=np.int8)
    sys.stdout = Logger(r"test.log")
    # 设置参数
    dataset={}
    dataset[0]="sps8_len512_num1000_train_ori.h5"
    dataset[1]="highsnr_sps4_len512_num10000_train_ori.h5"
    dataset[2]="highsnr_sps8_len512_num10000_train_ori.h5"
    # dataset[0]="highsnr_sps8_len128_num10000_train.h5"
    # dataset[1]="highsnr_sps4_len128_num10000_train.h5"
    k = 8
    args.class_num = 8
    args.max_iterations =2000
    args.test_interval = 1
    # args.max_iterations = 5
    # args.test_interval = 2
    args.lr=2e-3
    # 设置随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data_folder = r'E:\xuqiang\ch5_final\\'
    for i in list_ori_source:
    # 设置源域和目标域
        args.s_dset_path = data_folder +dataset[i]
        print(str(i)+" 此时源域是"+dataset[i])
        # list_ori_target=np.delete(list_ori_target,i)
        # list_ori_target=np.delete(list_ori_target,0)
        for j in [1]:
            print(str(i)+"_"+str(j)+" 此时目标域是"+dataset[j])
            args.t_dset_path = data_folder +dataset[j]
            # for k in [0,1,2,3,4,5]:
            for k in [5]:
                args.rate=k+1
                args.name=str(i+1)+"_"+str(j+1)+"_rate"+str(k+1)
                print(args.name)
                train(args)
                