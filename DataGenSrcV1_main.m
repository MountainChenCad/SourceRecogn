% 该数据集信号包含信道
% 信号类型包括["BPSK","QPSK","8PSK", "16QAM","32QAM","64QAM","CPFSK","GFSK","PAM4","PAM8"]
% 长度为256个点
clc;clear all;close all;
%定义信号种类
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%选择需要的信号种类%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modulationTypes = categorical(["BPSK","QPSK","8PSK",...
%     "16QAM","64QAM",...
%     "2FSK","GFSK",...
%     "PAM4"]);
modulationTypes = categorical(["BPSK","8PSK","16QAM","64QAM","PAM4","PAM8"]);
% modulationTypes = categorical(["2ASK"]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%定义参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numModulationTypes = length(modulationTypes);   %获取信号种类数目
numFramesPerModType = 1000;                      % 每种调制方式在每种信噪比下的样本个数
sps = 8;                                        % 每个符号的采样点
spf = 512;                                     % 单个样本长度（帧长度）
fs = 500e3;                                     % 采样率
first=0;                                       % 起始信噪比                     
last=30;                                         % 截止信噪比
foot=2;                                         % 信噪比跨度
snr_num=(last-first)/foot+1;                    % 信噪比总数
symbolsPerFrame = spf / sps;                    % 每一帧的符号数
co=0.5;
q=0;
%%%%%%%%%%创建一个在x方向无限维的数据,块大小为[1 2 spf],使用9级压缩%%%%%%%%%%
if numFramesPerModType>=500
    filename=["sps"+num2str(sps)+"_len"+num2str(spf)+"_num"+numFramesPerModType+"_train.h5"];
else
    filename=["sps"+num2str(sps)+"_len"+num2str(spf)+"_num"+numFramesPerModType+"_test.h5"];
end
% filename=["test.h5"];
% 创建H5文件
h5create(filename,'/ModData',[Inf 2 spf],'Datatype','double', ...
           'ChunkSize',[1 2 spf],'Deflate',9)
h5create(filename,'/ModType',[Inf numModulationTypes],'Datatype','int8', ...
           'ChunkSize',[1 numModulationTypes],'Deflate',9)
h5create(filename,'/Snr',[Inf],'Datatype','int8', ...
           'ChunkSize',[1],'Deflate',9)
%保存当前时间
channel=helperModClassTestChannel(...
    'SampleRate',fs,...
    'SNR',30,...
    'PathDelays',[0 1.8 3.4]/fs,...
    'AveragePathGains',[0 -2 -10],...
    'MaximumDopplerShift',4,...
    'KFactor',4,...
    'MaximumClockOffset',5,...
    'CenterFrequency',902e6);
tic 
%最终截取信号的偏移量
transDelay = 50;
index = numFramesPerModType*snr_num;
for modType = 1:numModulationTypes
    p=1;                               % 循环计数
    data=zeros(index,2,spf);
    one_hot_code1=int8(zeros(index,numModulationTypes));
    snr1=int8(zeros(index,1));
    %输出当前信号生成的信息,包括时间，产生的帧
    fprintf('%s - Generating %s frames\n', ...
      datestr(toc/86400,'HH:MM:SS'), modulationTypes(modType))
    %信号标签
    label = modulationTypes(modType);
    hotcode=int8(zeros(1,numModulationTypes));
    hotcode(modType)=1;
    %产生个数相同的数字符号
    dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
    %产生调制器
    modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs,co);
    for snr=linspace(first,last,snr_num)
        channel.SNR=snr;
        %开始产生信号
        for j=1:numFramesPerModType
          % 生成随机信号
          x = dataSrc();
          % 调制
          y = modulator(x);
          %rxSamples=awgn(y,snr,"measured");
          rxSamples = channel(y);
          frame = helperModClassFrameGenerator(rxSamples, spf, spf, transDelay, sps);
%           figure;
%           subplot(2,2,1);
%           plot(y);
%           subplot(2,2,2);
%           plot(abs(fftshift(fft((real(rxSamples)+imag(rxSamples))))));
%           subplot(2,2,3);
%           plot(abs(fftshift(fft((real(frame)+imag(frame))))));
%           subplot(2,2,4);
%           plot(frame);
          frame = frame.';
          Idata=real(frame);
          Qdata=imag(frame);
          ori_data=[Idata;Qdata];
          final=reshape(ori_data,[1 2 spf]);
          data(p,:,:)=final;        
          one_hot_code1(p,:)=hotcode;
          snr1(p)=int8(snr);
          p=p+1;
        end
    end
    startx = [1+q*index 1 1];
    countx = [index 2 spf];          
    starty = [1+q*index 1];
    county = [index numModulationTypes];           
    startz = 1+q*index;
    countz = index; 
    h5write(filename,'/ModData',data,startx,countx);
    h5write(filename,'/ModType',one_hot_code1,starty,county);
    h5write(filename,'/Snr',snr1,startz,countz);
    q=q+1;
end


