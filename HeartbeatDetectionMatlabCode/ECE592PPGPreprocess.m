close all;

Fs=125;
N=8;
FsN=Fs*N;
L=8;
FsL=Fs*L;
Fc=18;


testfiledir='Datasets\Training_data\';
datafiles=struct2cell(dir(fullfile(testfiledir,'*.mat')));
filenames=rot90(datafiles(1,:));
%find local mean and local standard deviation
for file_num=1:length(datafiles)
    file_num
    %load data and extract PPG rows
    file=strcat(testfiledir,filenames(file_num));
    patientdata=load(string(file));
    PPG1=patientdata.sig(2,:);
    PPG2=patientdata.sig(3,:);
    window=fix(length(PPG1)/FsN);
    remainder=rem(length(PPG1),FsN);
    %take nearest multiple of 8 elements in array, reshape
    %Window size FsN, Fs=sampling freq, N=8
    PPG1Window=reshape(PPG1(1:end-remainder),FsN,[]);
    PPG1Mean=mean(PPG1Window);
    PPG1StdDev=std(PPG1Window);
    PPG2Window=reshape(PPG2(1:end-remainder),FsN,[]);
    PPG2Mean=mean(PPG2Window);
    PPG2StdDev=std(PPG2Window);
    %Normalize PPGnorm=(PPGraw-uPPG)/(stddevPPG)
    PPG1Norm=zeros(1,length(PPG1)-remainder);
    PPG2Norm=PPG1Norm;

    for i=1:window
        PPG1Norm((i-1)*1000+1:i*(1000))=(PPG1((i-1)*1000+1:i*(1000))-PPG1Mean(i))/PPG1StdDev(i);
        PPG2Norm((i-1)*1000+1:i*(1000))=(PPG2((i-1)*1000+1:i*(1000))-PPG2Mean(i))/PPG2StdDev(i);
    end
    %moving average filter of length FsL, L=2
    PPG1Trend=movmean(PPG1Norm,FsL);
    PPG2Trend=movmean(PPG2Norm,FsL);
    %detrend signal
    PPG1Detrend=PPG1Norm-PPG1Trend;
    PPG2Detrend=PPG2Norm-PPG2Trend;
    %apply 7th order butterworth filter, Fc=18Hz
    [b,a] = butter(7,Fc/(Fs/2));
    PPG1_preproc=filter(b,a,PPG1Detrend);
    PPG2_preproc=filter(b,a,PPG2Detrend);
    
    %Talha - Adding ECG to the dataset
    ECG = patientdata.sig(1,1:length(PPG1_preproc));
    newSig = [ECG; PPG1_preproc; PPG2_preproc];
    
    preproc_filename1=strcat('PreprocessedTraining/PPG_Preproc_Pt',string(file_num));
    preproc_filename1=strcat(preproc_filename1,'.mat');
    save(preproc_filename1,'newSig');
end
figure
subplot(2,1,1)
plot(PPG1)
title('PPG')
xlabel('Index')
xlim([1 37000])
ylabel('ADC Output')
subplot(2,1,2)
plot(PPG1_preproc)
title('PPG after Preprocessing')
xlabel('Index')
xlim([1 37000])
ylabel('ADC Output')


