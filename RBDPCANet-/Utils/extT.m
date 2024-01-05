function [Xx_i, Xy_i] = extT( InImg ,PatchSize)
%UNTITLED 提取L1范数所需要的的样本集
%   此处显示详细说明
NumInput =size(InImg,2);
Xx_i=zeros(PatchSize,(PatchSize*NumInput));
Xy_i=zeros(PatchSize,(PatchSize*NumInput));
for i=1:NumInput
    Xx_i(:,(i*PatchSize-PatchSize+1):(i*PatchSize))=reshape(InImg(:,i),PatchSize,PatchSize); 
    Xy_i(:,(i*PatchSize-PatchSize+1):(i*PatchSize))=(Xx_i(:,(i*PatchSize-PatchSize+1):(i*PatchSize)))';
end
end

