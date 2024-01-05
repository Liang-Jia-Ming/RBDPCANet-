function [Xx_i, Xy_i] = extT( InImg ,PatchSize)
%UNTITLED ��ȡL1��������Ҫ�ĵ�������
%   �˴���ʾ��ϸ˵��
NumInput =size(InImg,2);
Xx_i=zeros(PatchSize,(PatchSize*NumInput));
Xy_i=zeros(PatchSize,(PatchSize*NumInput));
for i=1:NumInput
    Xx_i(:,(i*PatchSize-PatchSize+1):(i*PatchSize))=reshape(InImg(:,i),PatchSize,PatchSize); 
    Xy_i(:,(i*PatchSize-PatchSize+1):(i*PatchSize))=(Xx_i(:,(i*PatchSize-PatchSize+1):(i*PatchSize)))';
end
end

