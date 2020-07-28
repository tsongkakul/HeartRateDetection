close all;
clear all

Fs=125;
N=8;
FsN=Fs*N;
L=8;
FsL=Fs*L;
Fc=18;
segments = [];

%PCA
load('AllHBPreprocData\HB.mat');
[coeff,score,latent,~,explained] = pca(allHBsegments);
PCACoeff = coeff(:, 1:12);
% ComponentImportance = (latent/(sum(latent)))*100
% sum(ComponentImportance(1:12))

%sample locations 
testingfile='PreprocessedTest\TestingSegments.mat';
trainingfile='PreprocessedTraining\TrainingSegments.mat';
%load data and combine segments
load(testingfile);
testSegments = segments;
load(trainingfile);
segments = [segments; testSegments];
%define new data structure for all features
Lseg = length(segments);
features = zeros(Lseg, 303); %added one column for labeling
for segNum=1:Lseg
%for segNum=1:2 %for testing purposes
    segNum
    segment = segments(segNum,2:end); %remove the label for feature calculation
    len_seg = length(segment);
    %place the label in the features matrix
    features(segNum, 1) = segments(segNum,1);
    
    %The segment samples themselves (140 data pts)
    features(segNum, 2:141) = segment;
    %The mean and std of the segment (2 data pts)
    mean_seg = mean(segment);
    std_seg = std(segment);
    features(segNum,142) = mean_seg;
    features(segNum,143) = std_seg;
    %A three level quantized version of the segment, setting the values to
    %-1 where the segment amplitude are below the values mean-0.5sigma, to
    %0 when they are within the interval [u-0.5o, u+0.5o], and finally to
    %=1 when they are above u+0.5o. (140 data pts)
    quant_seg = segment;
    quant_seg(quant_seg<(mean_seg-0.5*std_seg)) = -1;
    quant_seg(quant_seg<=(mean_seg+0.5*std_seg) &quant_seg>=(mean_seg-0.5*std_seg)) = 0;
    quant_seg(quant_seg>(mean_seg+0.5*std_seg)) = 1;
    features(segNum,144:283) = quant_seg;
    %The 12 coefficients obtained projecting the segment along the 12
    %selected PCA eigenvectors (components) (12 data pts)
	PCAprojection = segment*PCACoeff;
    features(segNum,284:295) = PCAprojection;
    %The maximum and minimum intesity values of the segment. (2 data pts)
    [max_val,max_pos] = max(segment);
    [min_val,min_pos] = min(segment);
    features(segNum,296) = max_val;
    features(segNum,297) = min_val;
    %The position of the maximum and minimum intensity values within the
    %segment. (2 data pts)
    features(segNum,298) = max_pos;
    features(segNum,299) = min_pos;
    %The (time) difference between the locations corresponding to the
    %maximum and the minimum intensity values within the segment. (1 data pts) 
    features(segNum,300) = abs(max_pos-min_pos)/Fs;
    %The segment energy in three frequency bands: [0.04, 0.09], [0.09,
    %0.15], [0.15, 0.60] Hz. (3 data pts)
    N=2^16;
    FFT_seg=fft(segment,N);
    f=Fs/N*(0:N-1);  %frequency array for plotting
    energy_spec=(abs(FFT_seg)/Fs).^2; %energy spectral density is square of power spectral density
    features(segNum,301) = sum(energy_spec(22:48)); %0.04-0.09 range
    features(segNum,302) = sum(energy_spec(49:79)); %0.09-0.15 range
    features(segNum,303) = sum(energy_spec(80:315)); %0.15-0.60 range
end
figure
plot(segment)
figure
plot(quant_seg)
figure
plot(f,energy_spec)
seg_filename=strcat('Features.mat');
save(seg_filename,'features');
