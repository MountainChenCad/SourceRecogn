# 获取数据集
import numpy as np
import h5py
import matplotlib.pyplot as plt
from numpy.lib import index_tricks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.fftpack as fftpack


# def X2spc(X):
#     X_spc = []
#     for k in range(X.shape[0]):
#         I = X[k][0, :]
#         Q = X[k][1, :]
#         data_fft, data_fft2 = iq2spc(I, Q)    # 柯达
#         ap = np.array([data_fft, data_fft2])
#         X_spc.append(ap)
#     X_spc = np.array(X_spc)
#     return X_spc


def arr_iq2ap(X):
    X_ap = []
    for k in range(X.shape[0]):
        I = X[k][0, :]
        Q = X[k][1, :]
        amp, phase = iq2ampphase(I, Q)
        data_fft, data_fft2 = iq2spc(I, Q)
        ap = np.array([amp, phase, data_fft, data_fft2])
        # fre = np.diff(phase)  # 柯达，瞬时相位换成瞬时频率
        # fre = np.pad(fre, (0, 1))
        # data_fft, data_fft2 = iq2spc(I, Q)    # 柯达
        # ap = np.array([amp, fre])    # 柯达
        # ap = np.array([amp, phase])
        # ap = np.array([data_fft,data_fft2,data_fft4])

        X_ap.append(ap)
    X_ap = np.array(X_ap)
    return X_ap


def iq2ampphase(inphase, quad):
    amplitude = np.sqrt(np.square(inphase) + np.square(quad))
    amp_norm = np.linalg.norm(amplitude)  # L2 norm
    amplitude = amplitude / amp_norm  # normalise
    # phase = np.arctan(np.divide(quad, inphase))
    # phase = 2.*(phase - np.min(phase))/np.ptp(phase)-1 #rescale phase to range [-1, 1]
    phase = np.arctan2(quad, inphase)
    # if phase<0:
    #  phase=phase+2*math.pi;
    return amplitude, phase


def iq2spc(inphase, quad):
    iqdata = inphase + quad * 1j
    data_fft = abs(fftpack.fftshift(fftpack.fft(iqdata)))
    iqdata2 = iqdata ** 2
    data_fft2 = abs(fftpack.fftshift(fftpack.fft(iqdata2)))
    return data_fft, data_fft2


# 加载数据集函数
def get_sloader(dataset_name=None, trainlog=None):
    seed = 2021
    np.random.seed(seed)
    # 读取数据
    filename = dataset_name
    f = h5py.File(filename, 'r')
    A_train_ori = f['ModData'][:, ::]
    B_data = f['ModType'][::]
    f.close()
    A_train1 = A_train_ori.swapaxes(0, 2)
    B_data = B_data.swapaxes(0, 1)

    B_train = to_classes(B_data)
    # B_train = B_train.reshape(B_train.shape[0], 1)
    del B_data

    A_train = np.zeros((A_train1.shape[0], 6, A_train1.shape[2]), dtype=np.float32)     # 多通道多特征输入 柯达
    A_train[:, 0:2, :] = A_train1
    # del A_train_ori
    X_ap1 = arr_iq2ap(A_train1)  # 计算AP特征，频谱和平方谱特征
    # X_spc = X2spc(A_train1)  # 频谱信息  # MZK
    del A_train1
    A_train[:, 2:6, :] = X_ap1    # 多通道多特征输入 柯达
    # A_train[:, 4:6, :] = X_spc  # MZK

    # 提取特征
    # A_train = np.zeros((A_train_ori.shape[0],7,A_train_ori.shape[2]),dtype=np.float32)
    # A_train[:,0:2,:]=A_train_ori
    # del A_train_ori
    # X_ap1 = arr_iq2ap(A_train_ori)
    # A_train[:,2:,:]=X_ap1
    # 归一化
    scaler = MinMaxScaler((-1, 1))
    dimm = A_train.shape[2]
    dimm1 = A_train.shape[1]
    A_train = scaler.fit_transform(A_train.reshape(-1, dimm).T)
    A_train = A_train.T.reshape(-1, dimm1, dimm)
    # 输出数据维度
    with open(trainlog, 'a', encoding='utf-8') as f:
        f.write("\n数据集ModData维度：" + str(A_train.shape))
        f.write("\n数据集ModType维度：" + str(B_train.shape))
    '''
    fig, ax = plt.subplots(figsize=(20,10))
    plt.axis('off')
    for i in range(16):
      ax = fig.add_subplot(4,4,i+1)
      amp = A_train[i*2000+1999][4,:]
      phase = A_train[i*2000+1999][5,:]
      ax.set_xlabel('sample points')
      ax.set_ylabel('I',color='tab:red')
      ax.plot(amp, color='tab:red', linewidth=0.7)
      ax.tick_params(axis='y', labelcolor='tab:red')
      ax2 = ax.twinx()
      ax2.set_ylabel('Q', color='tab:blue')
      ax2.plot(phase, color='tab:blue', linewidth=0.7)
      ax2.tick_params(axis='y', labelcolor='tab:blue')
      plt.title(params.classes[int(B_train[i*2000+1999])])
      plt.tight_layout()
    plt.show()
    '''
    # 处理数据格式
    # in_shp的维度[-1,1,2,length]
    in_shp = list(A_train.shape[1:])
    in_shp = [-1] + [1] + in_shp
    # 设置X训练集
    # A_train=A_train.swapaxes(1,2)
    A_train1 = torch.tensor(A_train, dtype=torch.float32)
    A_train = torch.tensor(np.reshape(A_train1, in_shp), dtype=torch.float32)  # .tensor改为.as_tensor
    del A_train1
    # 设置Y训练集
    B_train = torch.tensor(B_train, dtype=torch.long)
    # 设置Z训练集
    # 设置训练数据加载器
    train_dataset = TensorDataset(A_train, B_train)
    # train_loader = DataLoader(train_dataset, batch_size= params.batch_size, shuffle= True)
    # return train_loader
    return train_dataset


def get_tloader(dataset_name=None, trainlog=None):
    seed = 2021
    np.random.seed(seed)
    # 读取数据
    filename = dataset_name
    f = h5py.File(filename, 'r')
    A_train_ori = f['ModData'][:, ::]
    A_train_ori = A_train_ori.swapaxes(0, 2)
    index = A_train_ori.shape[0]
    A_train1 = A_train_ori[0:index, ::]
    del A_train_ori
    B_data = f['ModType'][:, 0:index]
    B_data = B_data.swapaxes(0, 1)
    f.close()
    # A_train=A_train_ori.swapaxes(0,2)

    # B_data=B_data.swapaxes(0,1)
    B_train = to_classes(B_data)
    # B_train = B_train.reshape(B_train.shape[0], 1)
    del B_data

    # 提取特征
    A_train = np.zeros((A_train1.shape[0], 6, A_train1.shape[2]), dtype=np.float32) # 柯达
    A_train[:, 0:2, :] = A_train1
    # del A_train_ori
    X_ap1 = arr_iq2ap(A_train1)
    del A_train1
    A_train[:, 2:6, :] = X_ap1  # 柯达
    # A_train = np.zeros((A_train_ori.shape[0],7,A_train_ori.shape[2]),dtype=np.float32)
    # A_train[:,0:2,:]=A_train_ori
    # del A_train_ori
    # X_ap1 = arr_iq2ap(A_train_ori)
    # A_train[:,2:,:]=X_ap1
    # 归一化
    scaler = MinMaxScaler((-1, 1))
    dimm = A_train.shape[2]
    dimm1 = A_train.shape[1]
    A_train = scaler.fit_transform(A_train.reshape(-1, dimm).T)
    A_train = A_train.T.reshape(-1, dimm1, dimm)
    # 输出数据维度
    with open(trainlog, 'a', encoding='utf-8') as f:
        f.write("\n数据集ModData维度：" + str(A_train.shape))
        f.write("\n数据集ModType维度：" + str(B_train.shape))
    '''
    fig, ax = plt.subplots(figsize=(20,10))
    plt.axis('off')
    for i in range(16):
      ax = fig.add_subplot(4,4,i+1)
      amp = A_train[i*2000+1999][4,:]
      phase = A_train[i*2000+1999][5,:]
      ax.set_xlabel('sample points')
      ax.set_ylabel('I',color='tab:red')
      ax.plot(amp, color='tab:red', linewidth=0.7)
      ax.tick_params(axis='y', labelcolor='tab:red')
      ax2 = ax.twinx()
      ax2.set_ylabel('Q', color='tab:blue')
      ax2.plot(phase, color='tab:blue', linewidth=0.7)
      ax2.tick_params(axis='y', labelcolor='tab:blue')
      plt.title(params.classes[int(B_train[i*2000+1999])])
      plt.tight_layout()
    plt.show()
    '''
    # 处理数据格式
    # in_shp的维度[-1,1,2,length]
    in_shp = list(A_train.shape[1:])
    in_shp = [-1] + [1] + in_shp
    # 设置X训练集
    # A_train=A_train.swapaxes(1,2)
    A_train1 = torch.tensor(A_train, dtype=torch.float32)
    A_train = torch.tensor(np.reshape(A_train1, in_shp), dtype=torch.float32)  # .tensor改为.as_tensor
    del A_train1
    # 设置Y训练集
    B_train = torch.tensor(B_train, dtype=torch.long)
    # 设置Z训练集
    # 设置训练数据加载器
    train_dataset = TensorDataset(A_train, B_train)
    # train_loader = DataLoader(train_dataset, batch_size= params.batch_size, shuffle= True)
    # return train_loader
    return train_dataset


def to_classes(onehot):
    num_class = np.zeros(onehot.shape[0])
    for i in range(onehot.shape[0]):
        num_class[i] = np.where(onehot[i, :] == max(onehot[i, :]))[0]
    return num_class
