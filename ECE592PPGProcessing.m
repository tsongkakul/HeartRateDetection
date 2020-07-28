close all;

Fs=125;
N=8;
FsN=Fs*N;
L=8;
FsL=Fs*L;
Fc=18;

s1=load('G:\My Drive\Data Science Project\Datasets\TestData\TEST_S01_T01.mat');
PPG1=s1.sig(2,:);

%find local mean and local standard deviation
window=fix(length(PPG1)/FsN);
remainder=rem(length(PPG1),FsN);
%take nearest multiple of 8 elements in array, reshape
%Window size FsN, Fs=sampling freq, N=8
PPG1Window=reshape(PPG1(1:end-remainder),FsN,[]);
PPG1Mean=mean(PPG1Window);
PPG1StdDev=std(PPG1Window);

%Normalize PPGnorm=(PPGraw-uPPG)/(stddevPPG)
PPGNorm=zeros(1,length(PPG1)-remainder);

for i=1:window
    PPGNorm((i-1)*1000+1:i*(1000))=(PPG1((i-1)*1000+1:i*(1000))-PPG1Mean(i))/PPG1StdDev(i);
end
%moving average filter of length FsL, L=2
PPGTrend=movmean(PPGNorm,FsL);
%detrend signal
PPGDetrend=PPGNorm-PPGTrend;
%apply 7th order butterworth filter, Fc=18Hz
[b,a] = butter(7,Fc/(Fs/2));
PPG1_preproc=filter(b,a,PPGDetrend);
figure;
subplot(2,1,1)
plot(PPG1)
subplot(2,1,2)
plot(PPG1_preproc);

figure;
subplot(2,1,1)
plot(PPG1(1:1000))
subplot(2,1,2)
plot(PPG1_preproc(1:1000));
%% Ground Truth Extraction
%detect Rwaves with Pan-Tompkins

Fs=125;
N=8;
FsN=Fs*N;
L=8;

sample = 'DATA_01_TYPE01';
sampleName = strrep(sample, '_', '-');
DataFileName = fullfile('Datasets', 'Training_data', [sample, '.mat']);
s1 = load(DataFileName);
ECG1 = s1.sig(1,:);
figure;
[qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(ECG1,Fs,1);

% Compare Pan-Tompkin to BPMtrace 
%   BPM value in every 8-second time window with overlap by 6 seconds
%   buffer(x,n,p) divides the signal into segments = buffer(ECG1,8*Fs, 6*Fs)

%% Extract HB or NHB segments using Ri, the R-Peak locations
%   Find segment bounds and labels
%   HB signal segments extracted from Ri and resized to lseg
%   NHB signals extracted and resized considering R+.5delR

givenSignal = ECG1;
segmentRatio = 1.00;
lseg = 140;

%Mid Pt of HB
Ri = qrs_i_raw;
%Finding the difference between each midpoint
DelRi = diff(Ri);
%Mid Pt of NHB
RiNHB = round(DelRi/2)+Ri(1:end-1);
HBsegments = []; 
for i = 1:length(RiNHB)-1
    seg = givenSignal(RiNHB(i):RiNHB(i+1)); 
    segNew = imresize(seg, [1 lseg]); 
    HBsegments = [HBsegments; segNew];
end
NHBsegments = [];
for i = 1:length(Ri)-1
    seg = givenSignal(Ri(i):Ri(i+1));
    segNew = imresize(seg, [1 lseg]); 
    NHBsegments = [NHBsegments; segNew];
end


%% feature extraction
%segment samples
%mean and standard dev
%3 level quantized
%12 coefficient PCA
%max and min intesity
%position of max and min intensity
%time difference between max and min intensity
%energy in 3 frequency bands [0.04,0.09],[0.09,0.15]and[0.15,0.60]Hz

