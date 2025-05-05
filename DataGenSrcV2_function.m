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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ѡ����Ҫ���ź�����%%%%%%%%%%%%%%%%%%%%%%%%%%%;
% modulationTypes = categorical(["BPSK","8PSK","16QAM","64QAM","PAM4","PAM8"]);
modulationTypes = categorical(["8PSK","16QAM"]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�Զ������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numFramesPerModType = 2;                       % ÿ�ֵ��Ʒ�ʽ��ÿ��������µ���������
sps = 8;                                       % ÿ�����ŵĲ�����
spf = 512;                                     % �����������ȣ�֡���ȣ�
fs = 500e3;                                    % ������
fc=902e6                                       % ����Ƶ��
first=0;                                       % ��ʼ�����                     
last=30;                                       % ��ֹ�����
KFactor=4                                      % ��˹�ŵ�K����
foot=2;                                        % ����ȿ��
MaximumDopplerShift=4;
MaximumClockOffset=5;                          % ʱ��ƫ������                                                     
PathDelays=[0 1.8 3.4];                        % �ྶʱ��
AveragePathGains=[0 -2 -10];                   % �ྶ����
filepath='C:\Users\xq\Desktop\��ҵ����\������\software1\test_data\';
filename_ori='Sps4Len128Fs200e3Fc902e6Num1000Snr[0,2,30]Complex';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�ɱ����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numModulationTypes = length(modulationTypes);    % ��ȡ�ź�������Ŀ
filename=filepath+"\"+filename_ori+".h5";
snr_num=(last-first)/foot+1;                     % ���������
co=0.5;
q=0;
%%%%%%%%%%����ļ��Ƿ���ڣ�������������ļ���%%%%%%%%%%
if ~exist('filename','file')~=0
   filename_ori=filepath+"\"+filename_ori+datestr(clock,'yyyy-mm-dd-HH-MM-SS' )+".h5";
   filename=filename_ori+".h5";
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%����H5�ļ�%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h5create(filename,'/ModData',[Inf 2 spf],'Datatype','double', ...
           'ChunkSize',[1 2 spf],'Deflate',9)
h5create(filename,'/ModType',[Inf numModulationTypes],'Datatype','int8', ...
           'ChunkSize',[1 numModulationTypes],'Deflate',9)
h5create(filename,'/Snr',[Inf],'Datatype','int8', ...
           'ChunkSize',[1],'Deflate',9)
fid=fopen(filename_ori+".txt",'w');
fprintf(fid,"%s��%s","�����ź�����",modulationTypes);
fprintf(fid,"\n%s��%s\n","������Դ","��������");
fprintf(fid,"%s��%d\n","������/�ź�����/�����",numFramesPerModType);
fprintf(fid,"%s��%d\n","������������",spf);
fprintf(fid,"%s��%d\n","��ʼ�����(dB)",first);
fprintf(fid,"%s��%d\n","��ֹ�����(dB)",last);
fprintf(fid,"%s��%d\n","����Ȳ���(dB)",foot);
fprintf(fid,"%s��%d\n","����Ƶ��(Hz)",fs);
fprintf(fid,"%s��%s\n","�ŵ�����","�����ŵ�");
fprintf(fid,"%s��%d\n","ʱ��ƫ������",MaximumClockOffset);
fprintf(fid,"%s��%d\n","��˹�ŵ�K����",KFactor);
fprintf(fid,"%s��%d\n","�ྶʱ��",PathDelays);
fprintf(fid,"%s��%d\n","�ྶ����",AveragePathGains);
fprintf(fid,"%s��%d\n","��������ƫ��",MaximumDopplerShift);
fprintf(fid,"%s��%d\n","sps",sps);
fprintf(fid,"%s��%d\n","����Ƶ��(Hz)",fc);
% fclose(fid);
%�����ŵ�
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
% end


