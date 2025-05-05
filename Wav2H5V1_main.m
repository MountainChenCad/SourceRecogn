% function ans=DataGenSrcV2-function(spf,numFramesPerModType,real_data_path,filepath,filename_ori)
%modulationTypes = categorical(["BPSK","8PSK","16QAM","64QAM","PAM4","PAM8"]);
%numModulationTypes = length(modulationTypes);
clc;clear all;
spf = 128;
numFramesPerModType = 10;
real_data_path="E:\xuqiang\ch5_final\ch5datav2";
filepath='E:\xuqiang\filesavepath';
filename_ori='RealDataLen512Num1';

dirOutput = dir(fullfile(real_data_path,'*.wav')); % 遍历文件夹下所有.wav文件
fileNames = {dirOutput.name}; % 得到带后缀的文件名cell
modulationTypes = categorical(fileNames);
numModulationTypes = length(modulationTypes);
filename=filepath+"\"+filename_ori+".h5";
if ~exist('filename','file')~=0
   filename=filepath+"\"+filename_ori+datestr(clock,'yyyy-mm-dd-HH-MM-SS' )+".h5";
end
h5create(filename,'/ModData',[Inf 2 spf],'Datatype','double', ...
           'ChunkSize',[1 2 spf],'Deflate',9)
h5create(filename,'/ModType',[Inf numModulationTypes],'Datatype','int8', ...
           'ChunkSize',[1 numModulationTypes],'Deflate',9)
tic 
for modType = 1:numModulationTypes
    %["G:\ch5data\"+snr+"_sps"+num2str(sps)+"_"+char(modulationTypes(modType))+".wav"]
    fprintf('%s - Generating %s frames\n', ...
      datestr(toc/86400,'HH:MM:SS'), modulationTypes(modType))
    modvec_indx = 1;
    [y,fs]=audioread(real_data_path+"\"+string(modulationTypes(modType)));
    idata=y(:,1);
    qdata=y(:,2);
    raw_output_vector=idata+1j*qdata;
    sampler_indx = round(50+450*rand());
    mean(abs(raw_output_vector).^2);
    dataset=zeros(numFramesPerModType,2,spf);
    typeset=int8(zeros(numFramesPerModType,numModulationTypes));
	while sampler_indx + spf < length(raw_output_vector) && modvec_indx <= numFramesPerModType
        sampled_vector = raw_output_vector(sampler_indx:sampler_indx+spf-1);
        %Normalize the energy in this vector to be 1
        framePower = mean(abs(sampled_vector).^2);
        sampled_vector = sampled_vector / sqrt(framePower);
        %energy = sum(abs(sampled_vector))
        %sampled_vector=sampled_vector/energy;
        sampled_vector=sampled_vector.';
        dataset(modvec_indx,1,:) = real(sampled_vector);
        dataset(modvec_indx,2,:)= imag(sampled_vector);
        typeset(modvec_indx,modType)=1;
        %bound the upper end very high so it's likely we get multiple passes through
        %independent channels
        sampler_indx = sampler_indx+round(spf, round(length(raw_output_vector)*.05));
        modvec_indx = modvec_indx+1;
    end
    startx = [1+(modType-1)*numFramesPerModType 1 1];
    countx = [numFramesPerModType 2 spf];
    starty = [1+(modType-1)*numFramesPerModType 1];
    county = [numFramesPerModType numModulationTypes];  
    h5write(filename,'/ModData',dataset,startx,countx);
    h5write(filename,'/ModType',typeset,starty,county);
end



