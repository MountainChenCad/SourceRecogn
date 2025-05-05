clc;clear all;close all;
%modulationTypes = categorical(["QPSK"]);
modulationTypes = categorical(["BPSK","8PSK","16QAM","64QAM","PAM4","PAM8"]);
numModulationTypes = length(modulationTypes)-1;   
ModData=h5read("E:\xuqiang\code\dataset-master-matlab\sps8_len512_num1000_train.h5","/ModData");
ModType=h5read("E:\xuqiang\code\dataset-master-matlab\sps8_len512_num1000_train.h5","/ModType");
index=2;
len=512;
len1=16000;
dim=2
figure;
title("频谱");
f=(0:len-1)*(500000/len)-500000/2;
for i=0:numModulationTypes
    data=ModData(index+i*len1,:,:);
    data=reshape(data,[dim len]);
    com_data=data(1,:)+1j*data(2,:);
    datafft=abs(fftshift(fft(com_data)));
    subplot(3,5,i+1);
    plot(f,datafft);
    title(char(modulationTypes(i+1)));
end
figure;
title("二次方谱")
for i=0:numModulationTypes
    data=ModData(index+i*len1,:,:);
    data=reshape(data,[dim len]);
    com_data=data(1,:)+1j*data(2,:);
    datafft=abs(fftshift(fft(com_data.^2)));
    subplot(3,5,i+1);
    plot(f,datafft);
    title(char(modulationTypes(i+1)));
end
figure;
title("四次方谱");
for i=0:numModulationTypes
    data=ModData(index+i*len1,:,:);
    data=reshape(data,[dim len]);
    com_data=data(1,:)+1j*data(2,:);
    datafft=abs(fftshift(fft(com_data.^4)));
    subplot(3,5,i+1);
    plot(f,datafft);
    title(char(modulationTypes(i+1)));
end




