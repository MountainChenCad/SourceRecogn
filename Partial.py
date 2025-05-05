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

warnings.filterwarnings("ignore")


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


type = sys.getfilesystemencoding()


# 最终测试结果

def testall(test_dataloader, trained_model, args):
    model = network.CNN1(args.class_num)
    model.load_state_dict(torch.load(trained_model))
    model.cuda()
    model.train(False)
    with torch.no_grad():
        num_classes = args.class_num
        # 启动网络
        corr_cnt = 0
        total_iter = 0
        conf = np.zeros([num_classes, num_classes])
        confnorm = np.zeros([num_classes, num_classes])
        for data in test_dataloader:
            # 设置超参数
            input, label = data
            input = Variable(input.cuda())

            # 计算类别分类正确的个数
            _, _, _, output = model(input)
            # output = class_classifier(feature_extractor(input))
            pred = output.data.max(1, keepdim=True)[1]
            label.cpu().numpy()
            for p, l in zip(pred, label):
                if (p == l):
                    corr_cnt += 1
                conf[l, p] = conf[l, p] + 1
                # snrconf[int((s+4)/2),l,p] = snrconf[int((s+4)/2),l,p]+1
                total_iter += 1
        print(args.name + " AccuracyOverall" + " " + format(100 * corr_cnt / total_iter, '.4f') + "%(" + str(
            corr_cnt) + "/" + str(total_iter) + ")")
        # 绘制总的混淆矩阵
        for i in range(0, num_classes):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        plot_confusion_matrix(confnorm, mod_labels=args.classes, title=args.name + "Confusion Matrix Overall Snr")

        # 输出各个调制方式的识别率
        for i in range(len(confnorm)):
            print(str(args.classes[i]) + " " + str(format(100 * confnorm[i, i], '.4f')) + "%(" + str(
                int(conf[i, i])) + "/" + str(int(np.sum(conf[i, :]))) + ")")


def matadd(mat=None):
    res = 0
    for q in range(len(mat)):
        res += mat[q][q]
    return res


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
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
            _, _, _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()
    return accuracy, hist_tar, mean_ent


def train(args):
    ## 准备数据，设置训练集和测试集的bs
    train_bs = args.batch_size

    dsets = {}
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n****************开始加载源域数据****************")
    dsets["source"] = get_sloader(args.s_dset_path, args.trainlog)
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n****************开始加载目标域数据****************")
    dsets["target"] = get_tloader(args.t_dset_path, args.trainlog)
    # dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())
    # dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())
    # dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())
    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)

    # 用什么作为测试集比较好？？!!!
    '''
    if "ResNet" in args.net:
        params = {"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.ResNetFc(**params)
    
    if "VGG" in args.net:
        params = {"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True, 'class_num': args.class_num}
        base_network = network.VGGFc(**params)'''
    # 定义特征提取器
    base_network = network.CNN1(args.class_num)
    base_network = base_network.cuda()

    # 定义域对抗器
    ad_net1 = network.AdversarialNetwork(base_network.output_num(), 128, args.max_iterations).cuda()
    ad_net2 = network.AdversarialNetwork(base_network.output_num(), 128, args.max_iterations).cuda()
    ad_net3 = network.AdversarialNetwork(base_network.output_num(), 128, args.max_iterations).cuda()
    # 获取网络参数列表
    optimizer = torch.optim.Adam(
        list(base_network.parameters()) + list(ad_net2.parameters()) + list(ad_net1.parameters()) + list(
            ad_net3.parameters()), lr=args.lr)
    # 设置学习率调整方式
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    # 类级别权重
    class_weight = None
    best_ent = 1000
    best_acc = 0
    total_epochs = args.max_iterations // args.test_interval
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n****************开始训练****************")
    begin_time = time()
    for i in range(args.max_iterations + 1):
        # 准备开始训练

        base_network.train(True)
        ad_net1.train(True)
        ad_net2.train(True)
        ad_net3.train(True)
        # 更新学习率
        scheduler.step()
        # optimizer = lr_scheduler(optimizer, i, **schedule_param)
        ##更新权重
        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            # 获取当前的类级权重并且评估当前模型obtain the class-level weight and evalute the current model
            base_network.train(False)
            temp_acc, class_weight, mean_ent = image_classification(dset_loaders, base_network)
            class_weight = class_weight.cuda().detach()

            temp = [round(i, 4) for i in class_weight.cpu().numpy().tolist()]
            log_str = str(temp)
            # args.out_file.write(log_str+"\n")
            # args.out_file.flush()
            # if mean_ent < best_ent:
            if best_acc < temp_acc:
                best_ent, best_acc = mean_ent, temp_acc
                best_model = base_network.state_dict()
                log_str = "iter: {:05d}, temp best precision: {:.5f}, mean_entropy: {:.5f}".format(i, temp_acc,
                                                                                                   mean_ent)
                # args.out_file.write(log_str+"\n")
                # args.out_file.flush()
                # print(log_str)
                with open(args.trainlog, 'a', encoding='utf-8') as f:
                    f.write("\n" + log_str)
                    f.write("\nSaving Model Weight...")
                torch.save(best_model, args.weigthpath + args.name + ".pt")
            else:
                log_str = "iter: {:05d}, precision: {:.5f}, mean_entropy: {:.5f}".format(i, temp_acc, mean_ent)
                # args.out_file.write(log_str+"\n")
                # args.out_file.flush()
                with open(args.trainlog, 'a', encoding='utf-8') as f:
                    f.write("\n" + log_str)

                    # 设置源域共享给目标域的样本数，每test_interval调整一次
        if i % args.test_interval == 0:
            if args.mu > 0:
                epoch = i // args.test_interval
                # 共享的样本个数，随着epoch的增长而减小
                len_share = int(max(0, (train_bs // args.mu) * (1 - epoch / total_epochs)))
            elif args.mu == 0:
                len_share = 0  # no augmentation
            else:
                len_share = int(train_bs // abs(args.mu))  # fixed augmentation
            log_str = "\n{}, iter: {:05d}, source/ target/ middle: {:02d} / {:02d} / {:02d}\n".format("training ", i,
                                                                                                      train_bs,
                                                                                                      train_bs,
                                                                                                      len_share)
            # args.out_file.write(log_str)
            # args.out_file.flush()
            with open(args.trainlog, 'a', encoding='utf-8') as f:
                f.write("\n" + log_str)
                # 设置共享的样本，随机选取len_share个
            dset_loaders["middle"] = None
            if not len_share == 0:
                dset_loaders["middle"] = DataLoader(dsets["source"], batch_size=len_share, shuffle=True,
                                                    num_workers=args.worker,
                                                    drop_last=True)
                iter_middle = iter(dset_loaders["middle"])

        # iter()将有迭代能力的对象转化成迭代器
        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])
        if dset_loaders["middle"] is not None and i % len(dset_loaders["middle"]) == 0:
            iter_middle = iter(dset_loaders["middle"])
        # 获取源域数据和标签，目标域数据
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        # 作用？？
        if class_weight is not None and args.weight_cls and class_weight[labels_source].sum() == 0:
            continue
        # 获取源域、目标域的特征和最后的输出
        features_source1, features_source2, outputs_source = base_network(inputs_source)  # features_source3
        features_target1, features_target2, outputs_target = base_network(inputs_target)  # features_target3
        # 获取共享样本的特征和输出
        if dset_loaders["middle"] is not None:
            inputs_middle, labels_middle = iter_middle.next()
            features_middle1, features_middle2, features_middle3, outputs_middle = base_network(inputs_middle.cuda())
            features1 = torch.cat((features_source1, features_target1, features_middle1), dim=0)
            features2 = torch.cat((features_source2, features_target2, features_middle2), dim=0)
            # features3 = torch.cat((features_source3, features_target3, features_middle3), dim=0)
            outputs = torch.cat((outputs_source, outputs_target, outputs_middle), dim=0)
        else:
            features1 = torch.cat((features_source1, features_target1), dim=0)
            features2 = torch.cat((features_source2, features_target2), dim=0)
            # features3 = torch.cat((features_source3, features_target3), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
        # 设置分类的样本权重
        cls_weight = torch.ones(outputs.size(0)).cuda()
        if class_weight is not None and args.weight_aug:
            cls_weight[0:train_bs] = class_weight[labels_source]
            if dset_loaders["middle"] is not None:
                cls_weight[2 * train_bs::] = class_weight[labels_middle]

        # 计算源域的（权重）交叉熵损失
        if class_weight is not None and args.weight_cls:
            src_ = torch.nn.CrossEntropyLoss(reduction='none')(outputs_source, labels_source)
            weight = class_weight[labels_source].detach()
            src_loss = torch.sum(weight * src_) / (1e-8 + torch.sum(weight).item())
        else:
            src_loss = torch.nn.CrossEntropyLoss()(outputs_source, labels_source)
        # 计算源域和目标域的域对抗权重
        softmax_out = torch.nn.Softmax(dim=1)(outputs)
        entropy = my_loss.Entropy(softmax_out)
        transfer_loss = my_loss.DANN(features1, ad_net1, entropy, network.calc_coeff(i, 1, 0, 10, args.max_iterations),
                                     cls_weight, len_share)
        transfer_loss += my_loss.DANN(features2, ad_net2, entropy, network.calc_coeff(i, 1, 0, 10, args.max_iterations),
                                      cls_weight, len_share)
        # transfer_loss += my_loss.DANN(features3, ad_net3, entropy, network.calc_coeff(i, 1, 0, 10, args.max_iterations),
        #                               cls_weight, len_share)
        # 获取交叉熵正则化损失
        softmax_tar_out = torch.nn.Softmax(dim=1)(outputs_target)
        tar_loss = torch.mean(my_loss.Entropy(softmax_tar_out))
        # 获取最终损失
        total_loss = src_loss + transfer_loss + args.ent_weight * tar_loss

        if args.cot_weight > 0:
            if class_weight is not None and args.weight_cls:
                cot_loss = my_loss.marginloss(outputs_source, labels_source, args.class_num, alpha=args.alpha,
                                              weight=class_weight[labels_source].detach())
            else:
                cot_loss = my_loss.marginloss(outputs_source, labels_source, args.class_num, alpha=args.alpha)
            total_loss += cot_loss * args.cot_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    # torch.save(best_model, os.path.join(args.output_dir, "best_model.pt"))

    log_str = 'Accuracy: ' + str(np.round(best_acc, 2)) + "\n" + 'Mean_ent: ' + str(np.round(best_ent, 3)) + '\n'
    # args.out_file.write(log_str)
    # args.out_file.flush()
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n" + log_str)
        # print(log_str)
    end_time = time()
    run_time = end_time - begin_time
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n****************训练结束****************" + "\n训练总耗时" + str(run_time) + "s")

    # model=args.name+"_best_model.pt"
    # testall(dset_loaders["target"],model,args)

    # return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation for AMC')
    # 设置源域数据集
    parser.add_argument('--s_dset_path', type=str, default=r"E:\xuqiang\ch5_final\sps8_len512_num1000_train_ori.h5")
    # 设置目标域数据集
    parser.add_argument('--t_dset_path', type=str,
                        default=r"E:\xuqiang\ch5_final\highsnr_sps4_len512_num10000_train_ori.h5")
    # 设置模型权重保存路径
    parser.add_argument('--weigthpath', type=str, default=r"E:\xuqiang\software2\model_weight\\")
    # 设置日志保存路径
    parser.add_argument('--trainlog_path', type=str, default=r"E:\xuqiang\software2\trainlog\\")
    # 设置模型权重和日志名称
    parser.add_argument('--name', type=str, default=r"Sps8Len512Fs500e3Fc902e6Num1Snr[0,2,30]Complex")
    # 设置随机种子
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    # 设置GPU
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # 设置bz大小
    parser.add_argument('--batch_size', type=int, default=400, help="batch_size")
    # 最大迭代次数
    parser.add_argument('--max_iterations', type=int, default=1, help="max iterations")
    # 设置wordker数量
    parser.add_argument('--worker', type=int, default=1, help="number of workers")
    # 设置测试间隔
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
    # 设置学习率
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    # 设置源域种类数
    parser.add_argument('--class_num', type=int, default=6, help="Source Num")
    # 初始的对齐大小，batch_size//mu
    parser.add_argument('--mu', type=int, default=0, help="init augmentation size = batch_size//mu")
    # 目标域条件熵系数
    parser.add_argument('--ent_weight', type=float, default=0)
    # 自适应不确定性抑制损失系数
    parser.add_argument('--cot_weight', type=float, default=0, choices=[0, 1, 5, 10])
    # 是否对adv使用类级别权重
    parser.add_argument('--weight_aug', type=bool, default=True)
    # 是否对cls使用类级别权重
    parser.add_argument('--weight_cls', type=bool, default=True)
    # ？
    parser.add_argument('--alpha', type=float, default=1)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # args.classes=["BPSK","8PSK","PAM4","PAM8","16QAM","64QAM"]
    # 设置随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.trainlog = args.trainlog_path + args.name + ".txt"
    # 创建指示训练进度的文件
    with open(args.trainlog, 'w', encoding='utf-8') as f:
        f.write("此时源域数据集是" + args.s_dset_path)
        f.write("\n此时目标域数据集是" + args.t_dset_path)
        f.write("\n此时权重文件名称是" + args.name)
    train(args)
