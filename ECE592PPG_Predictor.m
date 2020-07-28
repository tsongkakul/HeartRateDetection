%Loads the models and evaluates them on the training data
load('ClassicalModels.mat');
load('TrainingFeatures.mat');
Models = {LDAModel, KNNFModel, KNNMModel, TreeModel, SVMLModel, SVMCModel};
y = features(:,1);
X = features(:,2:end);
for i=1:length(Models)
    tic
    yfit = Models{i}.predictFcn(X);
    toc
    sum((yfit-y)/length(y))
end
