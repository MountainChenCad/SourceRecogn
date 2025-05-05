function modulator = helperModClassGetModulator(modType, sps, fs,co)
%helperModClassGetModulator Modulation function selector
%   MOD = helperModClassGetModulator(TYPE,SPS,FS) returns the modulator
%   function handle MOD based on TYPE. SPS is the number of samples per
%   symbol and FS is the sample rate.
%   
%   See also ModulationClassificationWithDeepLearningExample.

%   Copyright 2019 The MathWorks, Inc.

switch modType
  case "BPSK"
    modulator = @(x)bpskModulator(x,sps,co);
  case "QPSK"
    modulator = @(x)qpskModulator(x,sps,co);
  case "8PSK"
    modulator = @(x)psk8Modulator(x,sps,co);
  case "16PSK"
    modulator = @(x)psk16Modulator(x,sps,co);
  case "32PSK"
    modulator = @(x)psk32Modulator(x,sps,co);
  case "16QAM"
    modulator = @(x)qam16Modulator(x,sps,co);
   case "32QAM"
    modulator = @(x)qam32Modulator(x,sps,co);
  case "64QAM"
    modulator = @(x)qam64Modulator(x,sps,co);
   case "128QAM"
    modulator = @(x)qam128Modulator(x,sps,co);
  case "256QAM"
    modulator = @(x)qam256Modulator(x,sps,co);
  case "512QAM"
    modulator = @(x)qam512Modulator(x,sps,co);
  case "1024QAM"
    modulator = @(x)qam1024Modulator(x,sps,co);
  case "GFSK"
    modulator = @(x)gfskModulator(x,sps);
  case "2FSK"
    modulator = @(x)fsk2Modulator(x,sps);
  case "4FSK"
    modulator = @(x)fsk4Modulator(x,sps);
  case "PAM4"
    modulator = @(x)pam4Modulator(x,sps,co);
  case "PAM8"
    modulator = @(x)pam8Modulator(x,sps,co);
  case "PAM16"
    modulator = @(x)pam16Modulator(x,sps,co);
  case "PAM32"
    modulator = @(x)pam32Modulator(x,sps);
  case "16APSK"
    modulator = @(x)apsk16Modulator(x,sps,co);
  case "32APSK"
    modulator = @(x)apsk32Modulator(x,sps,co);
  case "B-FM"
    modulator = @(x)bfmModulator(x, fs);
  case "DSB-AM"
    modulator = @(x)dsbamModulator(x, fs);
  case "SSB-AM"
    modulator = @(x)ssbamModulator(x, fs);
  case "2ASK"
    modulator = @(x)ask2Modulator(x, sps,co);
end
end
function y =apsk16Modulator(x,sps,co)
M = [8 8];
radii = [0.5 1.5];
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
%调制
syms = apskmod(x,M,radii);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y =apsk32Modulator(x,sps,co)
M = [4 8 20];
radii = [0.3 0.7 1.2];
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
%调制
syms = apskmod(x,M,radii);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = ask2Modulator(x,sps,co)
for i=1:length(x)
    syms(i)=(x(i)+1)*sqrt(2/5);
end
persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
y = y';
end

function y = bpskModulator(x,sps,co)
%bpskModulator BPSK modulator with pulse shaping
%   Y = bpskModulator(X,SPS) BPSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 1]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = pskmod(x,2);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qpskModulator(x,sps,co)
%qpskModulator QPSK modulator with pulse shaping
%   Y = qpskModulator(X,SPS) QPSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = pskmod(x,4,pi/4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = psk8Modulator(x,sps,co)
%psk8Modulator 8-PSK modulator with pulse shaping
%   Y = psk8Modulator(X,SPS) 8-PSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 7]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = pskmod(x,8);
% syms = pskmod(x,8,pi/8);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = psk16Modulator(x,sps,co)
%psk8Modulator 8-PSK modulator with pulse shaping
%   Y = psk8Modulator(X,SPS) 8-PSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 7]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = pskmod(x,16);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = psk32Modulator(x,sps,co)
%psk8Modulator 8-PSK modulator with pulse shaping
%   Y = psk8Modulator(X,SPS) 8-PSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 7]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = pskmod(x,32);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = qam16Modulator(x,sps,co)
%qam16Modulator 16-QAM modulator with pulse shaping
%   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 15]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate and pulse shape
syms = qammod(x,16,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam32Modulator(x,sps,co)
%qam16Modulator 16-QAM modulator with pulse shaping
%   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 15]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate and pulse shape
syms = qammod(x,32,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam64Modulator(x,sps,co)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 63]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = qammod(x,64,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = qam128Modulator(x,sps,co)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 63]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = qammod(x,128,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = qam256Modulator(x,sps,co)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 63]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = qammod(x,256,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = qam512Modulator(x,sps,co)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 63]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = qammod(x,512,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = qam1024Modulator(x,sps,co)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 63]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
end
% Modulate
syms = qammod(x,1024,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = pam4Modulator(x,sps,co)
%pam4Modulator PAM4 modulator with pulse shaping
%   Y = pam4Modulator(X,SPS) PAM4 modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs amp
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
  amp = 1 / sqrt(mean(abs(pammod(0:3, 4)).^2));
end
% Modulate
syms = amp * pammod(x,4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = pam8Modulator(x,sps,co)
%pam4Modulator PAM4 modulator with pulse shaping
%   Y = pam4Modulator(X,SPS) PAM4 modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs amp
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
  amp = 1 / sqrt(mean(abs(pammod(0:7, 8)).^2));
end
% Modulate
syms = amp * pammod(x,8);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = pam16Modulator(x,sps,co)
%pam4Modulator PAM4 modulator with pulse shaping
%   Y = pam4Modulator(X,SPS) PAM4 modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs amp
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(co, 6, sps);
  amp = 1 / sqrt(mean(abs(pammod(0:15, 16)).^2));
end
% Modulate
syms = amp * pammod(x,16);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = pam32Modulator(x,sps)
%pam4Modulator PAM4 modulator with pulse shaping
%   Y = pam4Modulator(X,SPS) PAM4 modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs amp
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 6, sps);
  amp = 1 / sqrt(mean(abs(pammod(0:31, 32)).^2));
end
% Modulate
syms = amp * pammod(x,32);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end


function y = gfskModulator(x,sps)
%gfskModulator GFSK modulator
%   Y = gfskModulator(X,SPS) GFSK modulates the input X and returns the
%   signal Y. X must be a column vector of values in the set [0 1]. The
%   BT product is 0.35 and the modulation index is 1. The output signal
%   Y has unit power.

persistent mod meanM
if isempty(mod)
  M = 2;
  mod = comm.CPMModulator(...
    'ModulationOrder', M, ...
    'FrequencyPulse', 'Gaussian', ...
    'BandwidthTimeProduct', 0.35, ...
    'ModulationIndex', 1, ...
    'SamplesPerSymbol', sps);
  meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end

function y = fsk2Modulator(x,sps)
%cpfskModulator CPFSK modulator
%   Y = cpfskModulator(X,SPS) CPFSK modulates the input X and returns
%   the signal Y. X must be a column vector of values in the set [0 1].
%   the modulation index is 0.5. The output signal Y has unit power.

persistent mod meanM
if isempty(mod)
  M = 2;
  mod = comm.CPFSKModulator(...
    'ModulationOrder', M, ...
    'ModulationIndex', 0.5, ...
    'SamplesPerSymbol', sps);
  meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end
function y = fsk4Modulator(x,sps)
%cpfskModulator CPFSK modulator
%   Y = cpfskModulator(X,SPS) CPFSK modulates the input X and returns
%   the signal Y. X must be a column vector of values in the set [0 1].
%   the modulation index is 0.5. The output signal Y has unit power.

persistent mod meanM
if isempty(mod)
  M = 4;
  mod = comm.CPFSKModulator(...
    'ModulationOrder', M, ...
    'ModulationIndex', 0.5, ...
    'SamplesPerSymbol', sps);
  meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end

function y = bfmModulator(x,fs)
%bfmModulator Broadcast FM modulator
%   Y = bfmModulator(X,FS) broadcast FM modulates the input X and returns
%   the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The frequency deviation is 75 kHz
%   and the pre-emphasis filter time constant is 75 microseconds.

persistent mod
if isempty(mod)
  mod = comm.FMBroadcastModulator(...
    'AudioSampleRate', fs, ...
    'SampleRate', fs);
end
y = mod(x);
end

function y = dsbamModulator(x,fs)
%dsbamModulator Double sideband AM modulator
%   Y = dsbamModulator(X,FS) double sideband AM modulates the input X and
%   returns the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The IF frequency is 50 kHz.

y = ammod(x,50e3,fs);
end

function y = ssbamModulator(x,fs)
%ssbamModulator Single sideband AM modulator
%   Y = ssbamModulator(X,FS) single sideband AM modulates the input X and
%   returns the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The IF frequency is 50 kHz.

y = ssbmod(x,50e3,fs);
end