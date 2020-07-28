close all;
%Import training features
featurefile='TrainingFeatures.mat';
%load data and extract segments
load(featurefile);

ground_truth = features(:,1);
feature_data = features(1:31932,2:303);
%linear discriminant
linear_disc = fitcdiscr(feature_data,ground_truth);
%quadratic discriminant
quad_disc = fitcdiscr(feature_data,ground_truth,'DiscrimType','quadratic');

%K-NN, k=5

%K-NN, k=10

%random forest

