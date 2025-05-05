% function ans=dataset_gen(sps,spf,fs,fc,first,last,foot,MaximumClockOffset,KFactor,PathDelays,AveragePathGains,numFramesPerModType,filepath,filename_ori,flag)
% sps=str2num(sps);
% spf=str2num(spf);
% fs=char(fs);fs=str2num(fs);
% fc=char(fc);fc=str2num(fc);
% first=char(first);first=str2num(first);
% last=char(last);last=str2num(last);
% foot=char(foot);foot=str2num(foot);
% KFactor=char(KFactor);KFactor=str2num(KFactor);
% MaximumClockOffset=char(MaximumClockOffset);MaximumClockOffset=str2num(MaximumClockOffset);
% PathDelays=char(PathDelays);PathDelays=str2num(PathDelays);
% AveragePathGains=char(AveragePathGains);AveragePathGains=str2num(AveragePathGains);
% numFramesPerModType=char(numFramesPerModType);numFramesPerModType=str2num(numFramesPerModType);
% filepath=char(filepath);
% filename_ori=char(filename_ori);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%选择需要的信号种类%%%%%%%%%%%%%%%%%%%%%%%%%%%;
% modulationTypes = categorical(["BPSK","8PSK","16QAM","64QAM","PAM4","PAM8"]);
modulationTypes = categorical(["8PSK","16QAM"]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%自定义参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numFramesPerModType = 2;                       % 每种调制方式在每种信噪比下的样本个数
sps = 8;                                       % 每个符号的采样点
spf = 512;                                     % 单个样本长度（帧长度）
fs = 500e3;                                    % 采样率
fc=902e6                                       % 中心频点
first=0;                                       % 起始信噪比                     
last=30;                                       % 截止信噪比
KFactor=4                                      % 莱斯信道K因子
foot=2;                                        % 信噪比跨度
MaximumDopplerShift=4;
MaximumClockOffset=5;                          % 时钟偏移因子                                                     
PathDelays=[0 1.8 3.4];                        % 多径时延
AveragePathGains=[0 -2 -10];                   % 多径增益
filepath='C:\Users\xq\Desktop\毕业论文\第四章\software1\test_data\';
filename_ori='Sps4Len128Fs200e3Fc902e6Num1000Snr[0,2,30]Complex';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%可变参数%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numModulationTypes = length(modulationTypes);    % 获取信号种类数目
filename=filepath+"\"+filename_ori+".h5";
snr_num=(last-first)/foot+1;                     % 信噪比总数
co=0.5;
q=0;
%%%%%%%%%%检测文件是否存在，若存在则更改文件名%%%%%%%%%%
if ~exist('filename','file')~=0
   filename_ori=filepath+"\"+filename_ori+datestr(clock,'yyyy-mm-dd-HH-MM-SS' )+".h5";
   filename=filename_ori+".h5";
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%创建H5文件%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h5create(filename,'/ModData',[Inf 2 spf],'Datatype','double', ...
           'ChunkSize',[1 2 spf],'Deflate',9)
h5create(filename,'/ModType',[Inf numModulationTypes],'Datatype','int8', ...
           'ChunkSize',[1 numModulationTypes],'Deflate',9)
h5create(filename,'/Snr',[Inf],'Datatype','int8', ...
           'ChunkSize',[1],'Deflate',9)
fid=fopen(filename_ori+".txt",'w');
fprintf(fid,"%s：%s","调制信号类型",modulationTypes);
fprintf(fid,"\n%s：%s\n","数据来源","仿真数据");
fprintf(fid,"%s：%d\n","样本数/信号类型/信噪比",numFramesPerModType);
fprintf(fid,"%s：%d\n","单个样本长度",spf);
fprintf(fid,"%s：%d\n","起始信噪比(dB)",first);
fprintf(fid,"%s：%d\n","终止信噪比(dB)",last);
fprintf(fid,"%s：%d\n","信噪比步进(dB)",foot);
fprintf(fid,"%s：%d\n","采样频率(Hz)",fs);
fprintf(fid,"%s：%s\n","信道类型","复杂信道");
fprintf(fid,"%s：%d\n","时钟偏移因子",MaximumClockOffset);
fprintf(fid,"%s：%d\n","莱斯信道K因子",KFactor);
fprintf(fid,"%s：%d\n","多径时延",PathDelays);
fprintf(fid,"%s：%d\n","多径增益",AveragePathGains);
fprintf(fid,"%s：%d\n","最大多普勒偏移",MaximumDopplerShift);
fprintf(fid,"%s：%d\n","sps",sps);
fprintf(fid,"%s：%d\n","中心频率(Hz)",fc);
% fclose(fid);
%创建信道
channel=helperModClassTestChannel(...
    'SampleRate',fs,...
    'SNR',30,...
    'PathDelays',PathDelays/fs,...
    'AveragePathGains',AveragePathGains,...
    'MaximumDopplerShift',MaximumDopplerShift,...
    'KFactor',KFactor,...
    'MaximumClockOffset',MaximumClockOffset,...
    'CenterFrequency',fc);
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
% end


