import argparse
import os
import random
from time import *

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

import warnings
import my_loss
import network
from fine_tune import Dataset_build
from get_loader import get_sloader, get_tloader
from plot_confusion_matrix import plot_confusion_matrix

warnings.filterwarnings("ignore")


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


def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["target"])  # 挂起
        for i in range(len(loader['target'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, _, outputs = model(inputs)    # 柯达
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
    # 准备数据，设置训练集和测试集的bs（训练数量）
    train_bs = args.batch_size

    dsets = {}
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n****************开始加载源域训练数据****************")
    dsets["source"] = get_sloader(args.s_dset_path, args.trainlog)  # 读取数据 s_dset_path中储存的是训练集文件夹
    # dsets["source"] = Dataset_build(args.s_dset_path)
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n****************开始加载源域测试数据****************")
    dsets["target"] = get_tloader(args.t_dset_path, args.trainlog)  # 读取数据 t_dset_path中储存的是测试集文件夹
    # dsets["target"] = Dataset_build(args.t_dset_path)
    # dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train())
    # dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train())
    # dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test())
    dset_loaders = {}
    # DataLoader的参数
    # dataset: Dataset类， 决定数据从哪读取以及如何读取
    # batch_size: 批大小
    # num_works: 是否多进程读取机制
    # shuffle: 每个epoch是否乱序
    # drop_last: 当样本数不能被batch_size整除时， 是否舍弃最后一批数据
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)

    # 定义特征提取器
    base_network = network.CNN1(args.class_num)
    base_network = base_network.cuda()

    # 获取网络参数列表
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  实现Adam算法
    # 参数：
    # params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
    # lr (float, 可选) – 学习率（默认：1e-3）
    # betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
    # eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    # weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
    optimizer = torch.optim.Adam(list(base_network.parameters()), lr=args.lr)
    # 设置学习率调整方式
    # CosineAnnealingWarmRestarts
    # T_0就是初始restart的epoch数目，T_mult就是重启之后因子，即重启后T_0 = T_0 * T_mult
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    # 类级别权重
    class_weight = None
    best_ent = 1000
    best_acc = 0
    total_epochs = args.max_iterations // args.test_interval
    with open(args.trainlog, 'a', encoding='utf-8') as f:
        f.write("\n****************开始训练****************")
    # time()返回自1970年1月1日0点后过去的时间(单位:s)
    begin_time = time()
    for i in range(args.max_iterations + 1):  # 2001
        # 准备开始训练

        base_network.train(True)
        # 更新学习率
        scheduler.step()
        # 更新权重
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

                    # iter()将有迭代能力的对象转化成迭代器
        if i % len(dset_loaders["source"]) == 0:
            # 把dset_loaders["source"]变为一个列表，可用next逐步提出
            iter_source = iter(dset_loaders["source"])
        # 获取源域数据和标签，目标域数据
        inputs_source, labels_source = iter_source.next()
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        # 作用？？
        if class_weight is not None and args.weight_cls and class_weight[labels_source].sum() == 0:
            continue
        # 获取源域、目标域的特征和最后的输出
        features_source1, features_source2, outputs_source = base_network(inputs_source)    # 柯达
        # 获取共享样本的特征和输出
        total_loss = torch.nn.CrossEntropyLoss()(outputs_source, labels_source)

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
    parser.add_argument('--s_dset_path', type=str, default=r"E:\academic\Doctor\XQ_Project\12domain\train")
    # 设置目标域数据集
    parser.add_argument('--t_dset_path', type=str, default=r"E:\academic\Doctor\XQ_Project\12domain\test")
    # 设置模型权重保存路径
    parser.add_argument('--weightpath', type=str, default=r"E:\academic\Doctor\XQ_Project\\ModelWeight\\")
    # 设置日志保存路径
    parser.add_argument('--trainlog_path', type=str, default=r"E:\academic\Doctor\XQ_Project\\TrainLog\\")
    # 设置模型权重和日志名称
    parser.add_argument('--name', type=str, default=r"KD_HighSnr_sps4_Awgn_len128")
    # 设置随机种子
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    # 设置GPU
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # 设置bz大小
    parser.add_argument('--batch_size', type=int, default=500, help="batch_size")
    # 最大迭代次数
    parser.add_argument('--max_iterations', type=int, default=2000, help="max iterations")
    # 设置wordker数量
    parser.add_argument('--worker', type=int, default=1, help="number of workers")
    # 设置测试间隔
    parser.add_argument('--test_interval', type=int, default=10, help="interval of two continuous test phase")
    # 设置学习率
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    # 设置源域种类数
    parser.add_argument('--class_num', type=int, default=9, help="Source Num")
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

    train_dir = r'E:\academic\Doctor\XQ_Project\12domain\train'
    test_dir = r'E:\academic\Doctor\XQ_Project\12domain\test'
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)
    for file in zip(train_files, test_files):
        train_file = file[0]
        test_file = file[1]
        args.s_dset_path = os.path.join(train_dir, train_file)
        args.t_dset_path = os.path.join(test_dir, test_file)
        args.name = 'KD_IQ_AF_' + train_file.split('_num1200_')[0]
        # 创建指示训练进度的文件
        args.trainlog = args.trainlog_path + args.name + ".txt"
        with open(args.trainlog, 'w', encoding='utf-8') as f:
            f.write("此时源域训练数据是" + args.s_dset_path)
            f.write("\n此时源域测试数据是" + args.t_dset_path)
            f.write("\n此时权重文件名称是" + args.name)
        train(args)



    # for i in range(0, len(train_files)):
    #     for j in range(0, len(test_files)):
    #         train_file = train_files[i]
    #         test_file = test_files[j]
    #         args.s_dset_path = os.path.join(train_dir, train_file)
    #         args.t_dset_path = os.path.join(test_dir, test_file)
    #         args.name = train_dir+'to'+test_dir
    #         # 创建指示训练进度的文件
    #         args.trainlog = args.trainlog_path + args.name + ".txt"
    #         with open(args.trainlog, 'w', encoding='utf-8') as f:
    #             f.write("此时源域训练数据是" + args.s_dset_path)
    #             f.write("\n此时源域测试数据是" + args.t_dset_path)
    #             f.write("\n此时权重文件名称是" + args.name)
    #         train(args)