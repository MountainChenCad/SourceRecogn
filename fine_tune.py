import numpy as np
import h5py
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import fftpack
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from Resnet_alter import model_fine_tune
import torch
from sklearn.preprocessing import MinMaxScaler
import os


warnings.filterwarnings("ignore")

datasets_name_list = {}


# 选择源域、目标域文件
# train_path = str('E:\\academic\\Data\\MZK_Dataset\\Training_Dataset\\-12dB_len8192_sigsys11.h5')
# test_path = str('E:\\academic\\Data\\MZK_Dataset\\Test_Dataset\\-12dB_len8192_sigsys11.h5')
train_path = str('E:\\academic\\Doctor\\XQ_Project\\12domain\\train\\HighSnr_sps16_Rcian_len128_num1200_train.h5')
test_path = str('E:\\academic\\Doctor\\XQ_Project\\12domain\\test\\HighSnr_sps4_Awgn_len128_num400_test.h5')
# trainlog_path = str('E:\\academic\\Data\\trainlog\\IQ_-12dB_to_-12dB.txt')
# 选择训练日志保存路径
trainlog_path = str('Train_Result\\Fine_Tuning\\sps、信道不同\\HighSnr_sps16_Rcian_to_HighSnr_sps4_Awgn.txt')


# 把IQ序列转化为AP序列
def iq2ampphase(inphase, quad):
    amplitude = np.sqrt(np.square(inphase) + np.square(quad))
    amp_norm = np.linalg.norm(amplitude)  # L2 norm
    amplitude = amplitude / amp_norm  # normalise
    phase = np.arctan2(quad, inphase)
    return amplitude, phase


# 把IQ序列转化为频谱信息（频谱幅度和二次方谱）
def iq2spc(inphase, quad):
    iqdata = inphase + quad * 1j
    data_fft = abs(fftpack.fftshift(fftpack.fft(iqdata)))
    iqdata2 = iqdata ** 2
    data_fft2 = abs(fftpack.fftshift(fftpack.fft(iqdata2)))
    return data_fft, data_fft2


# 合成序列
def X2spc(X):
    X_spc = []
    for k in range(X.shape[0]):
        I = X[k][0, :]
        Q = X[k][1, :]
        data_fft, data_fft2 = iq2spc(I, Q)    # 柯达
        ap = np.array([data_fft, data_fft2])
        X_spc.append(ap)
    X_spc = np.array(X_spc)
    return X_spc


def arr_iq2ap(X):
    X_ap = []
    for k in range(X.shape[0]):
        I = X[k][0, :]
        Q = X[k][1, :]
        amp, phase = iq2ampphase(I, Q)
        # fre = np.diff(phase)
        # fre = np.pad(fre, (0, 1))
        ap = np.array([amp, phase])

        X_ap.append(ap)
    X_ap = np.array(X_ap)
    return X_ap


# 列表整形
def to_classes(onehot):
    num_class = np.zeros(onehot.shape[0])
    for i in range(onehot.shape[0]):
        num_class[i] = np.where(onehot[i, :] == max(onehot[i, :]))[0]
    return num_class


def to_classes_alter(onehot):
    num_class = np.zeros(onehot.shape[0])
    for i in range(onehot.shape[0]):
        num_class[i] = onehot[i]
    return num_class


# 搭建数据集
def Dataset_build(h5_file_path):
    f = h5py.File(h5_file_path, 'r')
    data = f['ModData'][:, ::]  # ModData Data_IQ
    label = f['ModType'][::]  # ModType DataType
    # data = f['Data_IQ'][:, ::]  # ModData Data_IQ
    # label = f['DataType'][::]  # ModType DataType
    f.close()
    # 把batch_size放在第一位
    data = data.swapaxes(0, 2)
    label = label.swapaxes(0, 1)  # 横轴纵轴互换
    # to_class把64800*9变为64800*1，不同位置的1变为从0、1、2、3、4、5、6、7、8，仍然表示9种信号
    label = to_classes(label)  # 使用旧录入数据集时注释掉
    # label = to_classes_alter(label)  # 使用新录入数据集时使用
    index = data.shape[0]
    data = data[0:index, ::]  # ？

    A_train = np.zeros((data.shape[0], 6, data.shape[2]), dtype=np.float32)  # 柯达
    # SPC_data = X2spc(data)  # 频谱信息  # MZK
    A_train[:, 0:2, :] = data
    # A_train[:, 0:2, :] = data
    AP_data = arr_iq2ap(data)  # AP序列
    A_train[:, 2:4, :] = AP_data
    SPC_data = X2spc(data)  # 频谱信息  # MZK
    A_train[:, 4:6, :] = SPC_data  # MZK
    # 归一化
    scaler = MinMaxScaler((-1, 1))
    dimm = A_train.shape[2]
    dimm1 = A_train.shape[1]
    A_train = scaler.fit_transform(A_train.reshape(-1, dimm).T)  # 把(64800,4,128)转化为(128,259200)
    A_train = A_train.T.reshape(-1, dimm1, dimm)  # 把(128,259200)转化为(64800,4,128),此时已完成归一化

    # 处理数据格式
    # in_shp的维度[-1,1,4,length]
    in_shp = list(A_train.shape[1:])
    in_shp = [-1] + [1] + in_shp

    # 设置X训练集
    A_train1 = torch.tensor(A_train, dtype=torch.float32)  # 把A_train变为tensor形式
    A_train = torch.tensor(np.reshape(A_train1, in_shp), dtype=torch.float32)  # .tensor为把A_train变为tensor并且维度为(64800,1,4,128) 改为.as_tensor
    del A_train1

    # 设置Y训练集
    B_train = torch.tensor(label, dtype=torch.long)  # 把label改为tensor格式，维度为(64800,1)

    train_dataset = TensorDataset(A_train, B_train)

    return train_dataset


train_dataset = Dataset_build(train_path)
test_dataset = Dataset_build(test_path)


# 训练部分代码
def main(title=None):
    # 预处理
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    with open(trainlog_path, 'w', encoding='utf-8') as f:
        f.write('源域文件路径为：'+train_path+'\n'+'目标域文件路径为'+test_path)
    # 加载数据集
    batch_size = 100
    train_data = train_dataset
    val_data = test_dataset
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # 加载模型
    num_class = 12
    model = model_fine_tune(class_num=num_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    model = model.cuda()

    # 超参数
    criterion = nn.CrossEntropyLoss()
    learning_rate = 2e-3
    epoch = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    out_dir = "checkpoints/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for epoch in range(0, epoch):  # 从此处继续训练
        with open(trainlog_path, 'a', encoding='utf-8') as f:
            f.write('\nEpoch: %d' % (epoch + 1)+'\n')
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # torch.size([batch_size, num_class])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            train_loss_list.append(loss.item())  # 把每个epoch的loss记录到一个list中
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            train_acc_list.append(100. * correct / total)  # 把每个epoch的acc记录到一个list中
            # 记录数据至指定txt文档
            with open(trainlog_path, 'a', encoding='utf-8') as f:
                f.write('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total)+'\n')
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))
        # get the ac with testdataset in each epoch

        with open(trainlog_path, 'a', encoding='utf-8') as f:
            f.write("Waiting Val..."+'\n')
        print('Waiting Val...')
        with torch.no_grad():
            correct = 0.0
            sum_loss = 0.0  # 由0.0改为[]
            total = 0.0
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)  # 计算val loss
                # optimizer.step()
                # if loss != 0:
                #     loss.backward()
                #     optimizer.step()
                # sum_loss.append(loss.item())
                sum_loss += loss.item()
                val_loss_list.append(float(loss))  # 把每个epoch的每个loss记录到一个list中

                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                acc = float(100 * correct / total)
                # val_acc_list.append(acc)  # 把每个epoch的acc记录到一个list中
                acc_val = 100 * correct / total
                output_loss = sum_loss / (batch_idx + 1)

            val_acc_list.append(float(acc_val))  # 把每个epoch的acc记录到一个list中
            # val_loss_list.append(float(output_loss))  # 把每个epoch的平均loss记录到一个list中
            # final_loss = np.mean(sum_loss)
            # val_acc_list.append(100 * correct / total)
            # val_loss_list.append(final_loss)
            # print('Val\'s ac is: %.3f%%' % max(val_acc_list))
            with open(trainlog_path, 'a', encoding='utf-8') as f:
                f.write('Val\'s ac is: %.3f%%' % (100 * correct / total)+'\n')
                f.write('Val\'s loss is: %3f' % output_loss+'\n')
            print('Val\'s ac is: %.3f%%' % (100 * correct / total))
            print('Val\'s loss is: %3f' % output_loss)
            # print(sum_loss)


            # acc_val = 100 * correct / total
            # val_acc_list.append(acc_val)

        torch.save(model.state_dict(), out_dir + "last.pt")
        if acc_val == max(val_acc_list):
            torch.save(model.state_dict(), out_dir + "best.pt")
            print("save epoch {} model".format(epoch))
            with open(trainlog_path, 'a', encoding='utf-8') as f:
                f.write("save epoch {} model".format(epoch))

    # loss曲线和准确率曲线

    # plt.subplot(221)
    # plt.title("train_acc")
    # plt.plot(train_acc_list)
    # plt.subplot(222)
    # plt.title("train_loss")
    # plt.plot(train_loss_list)
    # plt.subplot(223)
    # plt.title("val_acc")
    # plt.plot(val_acc_list)
    # plt.subplot(224)
    # plt.title("val_loss")
    # plt.plot(val_loss_list)
    # if title is not None:
    #     plt.savefig('E:\\academic\\Data\\trainlog\\传统深度学习\\'+title+'.png')
    # plt.show()


if __name__ == "__main__":
    # for i in range(-12, 21, 2):
    #     train_path = str('E:\\academic\\Data\\MZK_Dataset\\Training_Dataset\\0dB_len8192_sigsys11.h5')
    #     test_path = str('E:\\academic\\Data\\MZK_Dataset\\Validation_Dataset\\'+str(i)+'dB_len8192_sigsys11.h5')
    #     trainlog_path = str('E:\\academic\\Data\\trainlog\\传统深度学习\\tradition_0dB_to_'+str(i)+'dB.txt')
    #     train_dataset = Dataset_build(train_path)
    #     test_dataset = Dataset_build(test_path)
    #     main(title='tradition_0dB_to'+str(i)+'dB')
    main(title='IQ_-12dB_to_-12dB')