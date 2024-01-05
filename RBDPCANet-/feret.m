% ==== PCANet Demo =======
% T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, 
% "PCANet: A simple deep learning baseline for image classification?" 
% IEEE Trans. Image Processing, vol. 24, no. 12, pp. 5017-5032, Dec. 2015. 

% Tsung-Han Chan [chantsunghan@gmail.com]
% Please email me if you find bugs, or have suggestions or questions!
% ========================

clear all; close all; clc; 
addpath('./Utils');                %其他程序所在
addpath('./Liblinear');
make; 


ImgFormat = 'gray'; %'color' or 'gray'

num_people=20;
pretrain=1;
Imagesize=[80,80];
pretotal=7;
trainpath='E:\suxin\angleTest\train\';
testpath='E:\suxin\angleTest\test\';
TrnData=zeros(Imagesize(1)*Imagesize(2),num_people*pretrain);
TrnLabels=zeros(num_people*pretrain,1);
TestData=zeros(Imagesize(1)*Imagesize(2),num_people*(pretotal-pretrain));
TestLabels=zeros(num_people*(pretotal-pretrain),1);
for j=1:num_people
    for i=1:pretrain
        im=imread([trainpath,int2str(j),'\',int2str(i),'.png']);
       iim = rgb2gray(im);
        TrnData(:,i+(j-1)*pretrain)=reshape(iim,Imagesize(1)*Imagesize(2),1);
        TrnLabels(i+(j-1)*pretrain,:)=j;
    end
end
for j=1:num_people
    for i=1:pretotal-pretrain
        im=imread([trainpath,int2str(j),'\',int2str(i+pretrain),'.png']);
        iim = rgb2gray(im);
        TestData(:,i+(j-1)*(pretotal-pretrain))=reshape(iim,Imagesize(1)*Imagesize(2),1);
        TestLabels(i+(j-1)*(pretotal-pretrain),:)=j;
    end
end
%% Loading data from MNIST Basic (10000 training, 2000 validation, 50000 testing) 
% load mnist_basic data


% ===== Reshuffle重新采样 the training data =====
% Randnidx = randperm(size(mnist_train,1));    1返回行数；randperm随机打乱序列号
% mnist_train = mnist_train(Randnidx,:);           多少行的全部元素
% =======================================



% ==== Subsampling  the Training and Testing sets ============
% (comment out the following four lines for a complete test) 
% TrnData = TrnData(:,1:4:end);  % sample around 2500 training samples
% TrnLabels = TrnLabels(1:4:end); % 
% TestData = TestData(:,1:50:end);  % sample around 1000 test samples  
% TestLabels = TestLabels(1:50:end); 
% ===========================================================

nTestImg = length(TestLabels);

%% PCANet parameters (they should be funed based on validation set; i.e., ValData & ValLabel)
% We use the parameters in our IEEE TIP paper
PCANet.NumStages = 2;
PCANet.PatchSize = [9 9];
PCANet.NumFilters = [8 8];
PCANet.HistBlockSize = [7 7]; 
PCANet.BlkOverLapRatio = 0.5;
PCANet.Pyramid = [];

fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

%% PCANet Training with 10000 samples

fprintf('\n ====== PCANet Training ======= \n')
TrnData_ImgCell = mat2imgcell(TrnData,Imagesize(1),Imagesize(2),ImgFormat); % 将 trndata 中的列转换为单元格convert columns in TrnData to cells 
clear TrnData; 
tic;
[ftrain V BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,1); % Blkidx 用于学习分块的 dr 投影矩阵，例如 wpca BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabels, ftrain', '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
clear ftrain; 


%% PCANet Feature Extraction and Testing 

TestData_ImgCell = mat2imgcell(TestData,Imagesize(1),Imagesize(2),ImgFormat); % convert columns in TestData to cells 
clear TestData; 

fprintf('\n ====== PCANet Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 
for idx = 1:1:nTestImg
    
    ftest = PCANet_FeaExt(TestData_ImgCell(idx),V,PCANet); % extract a test feature using trained PCANet model 

    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(ftest'), models, '-q'); % label predictoin by libsvm
   
    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 0==mod(idx,nTestImg/100); 
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end 
    
    TestData_ImgCell{idx} = [];
    
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of PCANet, followed by a linear SVM classifier =====');
fprintf('\n     PCANet training time: %.2f secs.', PCANet_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);
