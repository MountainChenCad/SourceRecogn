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


# 未使用Logger
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
# 未使用testall，训练网络

def testall(test_dataloader, trained_model, args):
    """
    初始化模型: model = network.CNN1(args.class_num)

    torch.save(obj, f, pickle_module=<module '...'>, pickle_protocol=2)保存一个序列化（serialized）的目标到磁盘参数: obj:保存对象,
    f:类文件对象 (必须实现写和刷新)或一个保存文件名的字符串, pickle_modul:用于 pickling 元数据和对象的模块, pickle_protocol:指定 pickle protocal 可以覆盖默认参数
    保存整个模型：torch.save(model,'save.pt')    只保存训练好的权重：torch.save(model.state_dict(), 'save.pt')

    torch.load(f, map_location=None, pickle_module=<module 'pickle' from '...'>)用来加载模型
    参数： f: 类文件对象 (返回文件描述符)或一个保存文件名的字符串 map_location:一个函数或字典规定如何映射存储设备 pickle_module:用于 unpickling 元数据和对象的模块

    torch.nn.Module.load_state_dict(state_dict, strict=True) 用来加载模型参数。将 state_dict 中的 parameters 和 buffers 复制到此
    module 及其子节点中 参数: state_dict:保存 parameters 和 persistent buffers 的字典  strict: 可选，bool型。state_dict 中的 key 是否和
    model.state_dict() 返回的 key 一致。

    example:
    torch.save(model,'save.pt')
    model.load_state_dict(torch.load("save.pt"))  #model.load_state_dict()函数把加载的权重复制到模型的权重中去

    model.train,model.eval,由于batchnorm和dropout在训练和测试时的作用不同，前者(train)用于训练，后者(eval)用于测试
    """
    model = network.CNN1(args.class_num)
    model.load_state_dict(torch.load(trained_model))
    model.cuda()
    model.train(False)  # 等效于model.eval()
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
            _, _, _, output = model(input)  # return三个值  # MZK
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


# 未使用matadd
def matadd(mat=None):
    res = 0
    for q in range(len(mat)):
        res += mat[q][q]
    return res


# 未使用image_train
def image_train(resize_size=256, crop_size=224):
    """
    transforms.Compose():主要作用是串联多个图片变换的操作
    参数:
    Resize: 把给定的图片resize到given size
    RandomCrop: 在一个随机的位置进行裁剪
    RandomHorizontalFlip: 以0.5的概率水平翻转给定的PIL图像
    ToTensor: 把图像的灰度范围从[0,255]变换到[0,1]之间
    Normalize:把图像的灰度范围从[0,1]变换到[-1,1]之间
    """
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# 未使用image_test
def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# 计算准确率的函数
def image_classification(loader, model):
    start_test = True
    with torch.no_grad():  # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track梯度
        iter_test = iter(loader["target"])  # 生成迭代器
        for i in range(len(loader['target'])):
            data = iter_test.next()  # 使用next函数不断获取迭代器的下一个数据
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()  # 将指定模型加载到GPU上
            _, _, _, outputs = model(inputs)  # MZK
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                # torch.cat是将两个张量（tensor）拼接在一起，cat是concatenate的意思，即拼接，联系在一起
                # torch.cat((A,B),dim)，dim = 0时表示按列拼接，dim = 1表示按行拼接
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    # output = torch.max(input, dim)
    # 参数: input: 一个tensor  dim: max函数索引的维度，0是每列的最大值，1是每行的最大值
    # 返回值： 返回两个tensor，第一个tensor是每行的最大值，第二个tensor是每行最大值的索引
    # 通常第一个返回值(values)是不需要的，所以往往只需要提取第二个tensor，并转换为array格式
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100
    # 计算准确率
    # torch.squeeze 维度压缩，去除掉维度为1的dim
    mean_ent = torch.mean(my_loss.Entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()
    # 计算平均熵
    # torch.nn.Softmax(dim) dim = n时，则第n维和为1

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()
    # 计算权重
    return accuracy, hist_tar, mean_ent


#  训练主体
def train(args):
    # 准备数据，设置训练集和测试集的bs
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
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True)

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

    '''
    此处与深度学习不同，使用域对抗器来对特征进行提取
    '''
    # 定义域对抗器
    ad_net1 = network.AdversarialNetwork(base_network.output_num(), 128, args.max_iterations).cuda()
    ad_net2 = network.AdversarialNetwork(base_network.output_num(), 128, args.max_iterations).cuda()
    ad_net3 = network.AdversarialNetwork(base_network.output_num(), 128, args.max_iterations).cuda()
    # 获取网络参数列表
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  实现Adam算法
    # 参数：
    # params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
    # lr (float, 可选) – 学习率（默认：1e-3）
    # betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
    # eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
    # weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
    '相较于深度学习，迁移学习的优化器增加了三个域对抗器的参数'
    optimizer = torch.optim.Adam(
        list(base_network.parameters()) + list(ad_net2.parameters()) + list(ad_net1.parameters()) + list(
            ad_net3.parameters()), lr=args.lr)
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
        # 更新权重
        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            # test_interval 每进行n次迭代进行一次测试 相对的是caffe在训练过程中边训练边测试
            # 获取当前的类级权重并且评估当前模型obtain the class-level weight and evalute the current model
            base_network.train(False)
            temp_acc, class_weight, mean_ent = image_classification(dset_loaders, base_network)
            # 当前准确度，类别权重，平均熵
            # detach后得到的tensor仍指向原变量存放位置，但不具有grad
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
            if args.mu > 0:  # mu ？？
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
        features_source1, features_source2, features_source3, outputs_source = base_network(inputs_source)
        features_target1, features_target2, features_target3, outputs_target = base_network(inputs_target)
        # 获取共享样本的特征和输出
        if dset_loaders["middle"] is not None:
            inputs_middle, labels_middle = iter_middle.next()
            features_middle1, features_middle2, features_middle3, outputs_middle = base_network(inputs_middle.cuda())
            features1 = torch.cat((features_source1, features_target1, features_middle1), dim=0)
            features2 = torch.cat((features_source2, features_target2, features_middle2), dim=0)
            features3 = torch.cat((features_source3, features_target3, features_middle3), dim=0)  # MZK
            outputs = torch.cat((outputs_source, outputs_target, outputs_middle), dim=0)
        else:
            features1 = torch.cat((features_source1, features_target1), dim=0)
            features2 = torch.cat((features_source2, features_target2), dim=0)
            features3 = torch.cat((features_source3, features_target3), dim=0)  # MZK
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
        transfer_loss = my_loss.DANN(features1, ad_net1, None, network.calc_coeff(i, 1, 0, 10, args.max_iterations),
                                     cls_weight, len_share)
        transfer_loss += my_loss.DANN(features2, ad_net2, None, network.calc_coeff(i, 1, 0, 10, args.max_iterations),
                                      cls_weight, len_share)
        transfer_loss += my_loss.DANN(features3, ad_net3, None, network.calc_coeff(i, 1, 0, 10, args.max_iterations),
                                      cls_weight, len_share)
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
    # argparse是一个Python模块：命令行选项、参数和子命令解析器，使用 argparse 的第一步是创建一个 ArgumentParser 对象。
    # ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
    # parser = argparse.ArgumentParser(description='Process some integers.')
    parser = argparse.ArgumentParser(description='BA3US for Partial Domain Adaptation for AMC')
    # ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][,
    # required][, help][, metavar][, dest])
    # name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo。
    # action - 命令行遇到参数时的动作，默认值是 store。
    # store_const，表示赋值为const；
    # append，将遇到的值存储成列表，也就是如果参数重复则会保存多个值;
    # append_const，将参数规范中定义的一个值保存到一个列表；
    # count，存储遇到的次数；此外，也可以继承 argparse.Action 自定义参数解析；
    # nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号，当不指定值时对于 Positional argument 使用 default，对于
    # Optional argument 使用 const；或者是 * 号，表示 0 或多个参数；或者是 + 号表示 1 或多个参数。
    # const - action 和 nargs 所需要的常量值。
    # default - 不指定参数时的默认值。
    # type - 命令行参数应该被转换成的类型。
    # choices - 参数可允许的值的一个容器。
    # required - 可选参数是否可以省略 (仅针对可选参数)。
    # help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
    # metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
    # dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.
    # 设置源域数据集
    parser.add_argument('--s_dset_path', type=str,
                        default=r"E:\academic\Doctor\XQ_Project\12domain\train\LowSnr_sps4_Awgn_len128_num1200_train.h5")
    # 设置目标域数据集
    parser.add_argument('--t_dset_path', type=str,
                        default=r"E:\academic\Doctor\XQ_Project\12domain\test\LowSnr_sps8_Awgn_len128_num400_test.h5")
    # 设置模型权重保存路径
    parser.add_argument('--weigthpath', type=str, default=r"E:\PythonProject\New Code\ModelWeight\Non_Partial\sps不同\\")
    # 设置日志保存路径
    parser.add_argument('--trainlog_path', type=str, default=r"E:\PythonProject\New Code\Train_Result\Non_Partial\sps不同\\")
    # 设置模型权重和日志名称
    parser.add_argument('--name', type=str, default=r"LowSnr_sps4_Awgn_to_LowSnr_sps8_Awgn")
    # 设置随机种子
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    # 设置GPU
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # 设置bz大小
    parser.add_argument('--batch_size', type=int, default=400, help="batch_size")
    # 最大迭代次数
    parser.add_argument('--max_iterations', type=int, default=500, help="max iterations")
    # 设置wordker数量
    parser.add_argument('--worker', type=int, default=1, help="number of workers")
    # 设置测试间隔 单位？
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
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
    parser.add_argument('--weight_aug', type=bool, default=False)
    # 是否对cls使用类级别权重
    parser.add_argument('--weight_cls', type=bool, default=False)
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
    # args.trainlog = args.trainlog_path + args.name + ".txt"
    # # 创建指示训练进度的文件
    # with open(args.trainlog, 'w', encoding='utf-8') as f:
    #     f.write("此时源域数据集是" + args.s_dset_path)
    #     f.write("\n此时目标域数据集是" + args.t_dset_path)
    #     f.write("\n此时权重文件名称是" + args.name)
    # train(args)


    train_dir = r'E:\academic\Doctor\XQ_Project\12domain\train'
    test_dir = r'E:\academic\Doctor\XQ_Project\12domain\test'
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)
    for i in range(0, len(train_files)):
        for j in range(0, len(test_files)):
            train_file = train_files[i]
            test_file = test_files[j]
            args.s_dset_path = os.path.join(train_dir, train_file)
            args.t_dset_path = os.path.join(test_dir, test_file)
            args.name = train_dir+'to'+test_dir
            # 创建指示训练进度的文件
            args.trainlog = args.trainlog_path + args.name + ".txt"
            with open(args.trainlog, 'w', encoding='utf-8') as f:
                f.write("此时源域训练数据是" + args.s_dset_path)
                f.write("\n此时源域测试数据是" + args.t_dset_path)
                f.write("\n此时权重文件名称是" + args.name)
            train(args)
