% close all;

Fs=125;
N=8;
FsN=Fs*N;
L=8;
FsL=Fs*L;
Fc=18;
segments = [];
allHBsegments = [];

testfiledir='PreprocessedTraining\';
datafiles=struct2cell(dir(fullfile(testfiledir,'*.mat')));
filenames=rot90(datafiles(1,:));
%find local mean and local standard deviation
for file_num=1:length(datafiles)
    file_num
    %load data and extract PPG rows
    file=strcat(testfiledir,filenames(file_num));
    patientdata=load(string(file));
    ECG =patientdata.newSig(1,:);
    PPG1=patientdata.newSig(2,:);
    PPG2=patientdata.newSig(3,:);
    
    %Pan-Tompkins
    [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(ECG,Fs,0);
    
    % Extract HB or NHB segments using Ri, the R-Peak locations
    %   Find segment bounds and labels
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
        seg = PPG1(RiNHB(i):RiNHB(i+1)); 
        segNew = imresize(seg, [1 lseg]); 
        HBsegments = [HBsegments; segNew];
        seg = PPG2(RiNHB(i):RiNHB(i+1)); 
        segNew = imresize(seg, [1 lseg]); 
        HBsegments = [HBsegments; segNew];
    end
    NHBsegments = [];
    for i = 1:length(Ri)-1
        seg = PPG1(Ri(i):Ri(i+1));
        segNew = imresize(seg, [1 lseg]); 
        NHBsegments = [NHBsegments; segNew];
        seg = PPG2(Ri(i):Ri(i+1));
        segNew = imresize(seg, [1 lseg]); 
        NHBsegments = [NHBsegments; segNew];
    end
%     figure;plot(NHBsegments(1:6,:)')
%     figure;plot(HBsegments(1:6,:)')
    
    %Uncomment to test segmentation of ECG signals into HB and NHB regions
% %     HBsegments = []; 
% %     for i = 1:length(RiNHB)-1
% %         seg = ECG(RiNHB(i):RiNHB(i+1)); 
% %         segNew = imresize(seg, [1 lseg]); 
% %         HBsegments = [HBsegments; segNew];
% %     end
% %     NHBsegments = [];
% %     for i = 1:length(Ri)-1
% %         seg = ECG(Ri(i):Ri(i+1));
% %         segNew = imresize(seg, [1 lseg]); 
% %         NHBsegments = [NHBsegments; segNew];
% %     end
% %     
% %     figure;
% %     subplot(221);
% %     plot(HBsegments(1:20,:)')
% %     subplot(222);
% %     plot(NHBsegments(1:20,:)')
% %     
% %     HBsegments = []; 
% %     for i = 1:length(RiNHB)-1
% %         seg = PPG1(RiNHB(i):RiNHB(i+1)); 
% %         segNew = imresize(seg, [1 lseg]); 
% %         HBsegments = [HBsegments; segNew];
% %     end
% %     NHBsegments = [];
% %     for i = 1:length(Ri)-1
% %         seg = PPG1(Ri(i):Ri(i+1));
% %         segNew = imresize(seg, [1 lseg]); 
% %         NHBsegments = [NHBsegments; segNew];
% %     end
% %     subplot(223);
% %     plot(HBsegments(1:20,:)')
% %     subplot(224);
% %     plot(NHBsegments(1:20,:)')

    %Label and concatenate the data
    segments = [segments; [ones(length(HBsegments),1) HBsegments]];
    segments = [segments; [zeros(length(NHBsegments),1) NHBsegments]];
    allHBsegments = [allHBsegments; HBsegments];
end

% seg_filename=strcat('TestingSegments.mat');
% save(seg_filename,'segments');

% seg_filename=strcat('HB.mat');
% save(seg_filename,'allHBsegments');