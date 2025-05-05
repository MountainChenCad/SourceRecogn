% status=system('D:\Anaconda3\envs\torch\python.exe E:\xuqiang\ch5_final\software\print.py');
% status=system('D:\Anaconda3\envs\torch\python.exe run_partial.py');
% close all;clc;
% function cmd_py(python_inter)
%     python_inter
% python_inter="D:\Anaconda3\envs\torch\python.EXE";
% source="E:\xuqiang\ch5_final\sps8_len512_num1000_train_ori.h5";
% target="E:\xuqiang\ch5_final\highsnr_sps4_len512_num10000_train_ori.h5";
% weightpath="E:\xuqiang\software_final\ModelWeight\";
% trainlog_path="E:\xuqiang\software_final\TrainLog\";
% name="Sps8Len512Fs500e3Fc902e6Num1Snr[0,2,30]Complex";
% gpu_id=int16(0);
% seed=int16(2021);
% batch_size=int16(400);
% max_iterations=int16(2);
% worker=int16(1);
% test_interval=int16(1);
% lr=2e-3;
% class_num=int16(6);
% command=python_inter+" Non-Partial.py "+"--s_dset_path "+source+" --t_dset_path "+target+" --weigthpath "+weightpath+" --trainlog_path "+trainlog_path+ ...,
%     " --name "+name+" --gpu_id "+gpu_id+" --seed "+seed+" --batch_size "+batch_size+" --max_iterations "+max_iterations+" --worker "+worker+ ...,
%     " --test_interval "+test_interval+" --lr "+lr+" --class_num "+class_num;
% % command=python_inter+" Non-PartialV1_run_partial.py "+"--s_dset_path "+source+" --t_dset_path "+target+" --weigthpath "+weightpath+" --trainlog_path "+trainlog_path;
% 
% [status,result]=system(command);
% command=python_inter+" test.py ";
% command=python_inter+" Non-PartialV1_run_partial.py "+"--s_dset_path "+source+" --t_dset_path "+target+" --weigthpath "+weightpath+" --trainlog_path "+trainlog_path;
% target="E:\software\DataSet\highsnr_sps4_len512_num10000_train_ori_rate1.h5";
% name="E:\software\ModelWeight\MMDA_MR_1_2_rate1_best_model.pt";
% result_path="E:\software\Results";
% gpu_id=int16(0);
% batch_size=int16(400);
% worker=int16(1);
% classes="BPSK","8PSK","PAM4","PAM8","16QAM","64QAM";
function cmd_py(command)
% command=python_inter+" test.py "+" --t_dset_path "+target+" --name "+name+" --result_path "+result_path+ ...,
%     " --gpu_id "+gpu_id+" --batch_size "+batch_size+" --worker "+worker+ ...,
%     " --temp_classes "+classes;
[status,result]=system(command);
