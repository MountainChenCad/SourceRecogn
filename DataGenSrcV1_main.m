% �����ݼ��źŰ����ŵ�
% �ź����Ͱ���["BPSK","QPSK","8PSK", "16QAM","32QAM","64QAM","CPFSK","GFSK","PAM4","PAM8"]
% ����Ϊ256����
clc;clear all;close all;
%�����ź�����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ѡ����Ҫ���ź�����%%%%%%%%%%%%%%%%%%%%%%%%%%%
% modulationTypes = categorical(["BPSK","QPSK","8PSK",...
%     "16QAM","64QAM",...
%     "2FSK","GFSK",...
%     "PAM4"]);
modulationTypes = categorical(["BPSK","8PSK","16QAM","64QAM","PAM4","PAM8"]);
% modulationTypes = categorical(["2ASK"]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numModulationTypes = length(modulationTypes);   %��ȡ�ź�������Ŀ
numFramesPerModType = 1000;                      % ÿ�ֵ��Ʒ�ʽ��ÿ��������µ���������
sps = 8;                                        % ÿ�����ŵĲ�����
spf = 512;                                     % �����������ȣ�֡���ȣ�
fs = 500e3;                                     % ������
first=0;                                       % ��ʼ�����                     
last=30;                                         % ��ֹ�����
foot=2;                                         % ����ȿ��
snr_num=(last-first)/foot+1;                    % ���������
symbolsPerFrame = spf / sps;                    % ÿһ֡�ķ�����
co=0.5;
q=0;
%%%%%%%%%%����һ����x��������ά������,���СΪ[1 2 spf],ʹ��9��ѹ��%%%%%%%%%%
if numFramesPerModType>=500
    filename=["sps"+num2str(sps)+"_len"+num2str(spf)+"_num"+numFramesPerModType+"_train.h5"];
else
    filename=["sps"+num2str(sps)+"_len"+num2str(spf)+"_num"+numFramesPerModType+"_test.h5"];
end
% filename=["test.h5"];
% ����H5�ļ�
h5create(filename,'/ModData',[Inf 2 spf],'Datatype','double', ...
           'ChunkSize',[1 2 spf],'Deflate',9)
h5create(filename,'/ModType',[Inf numModulationTypes],'Datatype','int8', ...
           'ChunkSize',[1 numModulationTypes],'Deflate',9)
h5create(filename,'/Snr',[Inf],'Datatype','int8', ...
           'ChunkSize',[1],'Deflate',9)
%���浱ǰʱ��
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
%���ս�ȡ�źŵ�ƫ����
transDelay = 50;
index = numFramesPerModType*snr_num;
for modType = 1:numModulationTypes
    p=1;                               % ѭ������
    data=zeros(index,2,spf);
    one_hot_code1=int8(zeros(index,numModulationTypes));
    snr1=int8(zeros(index,1));
    %�����ǰ�ź����ɵ���Ϣ,����ʱ�䣬������֡
    fprintf('%s - Generating %s frames\n', ...
      datestr(toc/86400,'HH:MM:SS'), modulationTypes(modType))
    %�źű�ǩ
    label = modulationTypes(modType);
    hotcode=int8(zeros(1,numModulationTypes));
    hotcode(modType)=1;
    %����������ͬ�����ַ���
    dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
    %����������
    modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs,co);
    for snr=linspace(first,last,snr_num)
        channel.SNR=snr;
        %��ʼ�����ź�
        for j=1:numFramesPerModType
          % ��������ź�
          x = dataSrc();
          % ����
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


