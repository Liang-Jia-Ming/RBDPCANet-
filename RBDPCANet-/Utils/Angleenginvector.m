function [ V ] = Angleenginvector( XX,PatchSize,NumFilters )
%ANGLE 2DPCANET提取特征向量
%   此处显示详细说明
addpath('./Utils')
N=size(XX,2);
num=N/PatchSize;
V=zeros(PatchSize,NumFilters);
I=zeros(PatchSize,NumFilters);
VS=zeros(PatchSize,NumFilters);
H=zeros(PatchSize,NumFilters);
B=zeros(PatchSize,NumFilters);
mini = min(PatchSize, NumFilters); 
P=zeros(PatchSize,PatchSize);
G=zeros(PatchSize,PatchSize);
for i=1:mini
    V(i,i)=1;
    I(i,i)=1;
end
for i=1:num
    P=P+XX(:,1+(num-1)*PatchSize:num*PatchSize);
end
mean_A=P/num;
es=1;
while es
    VS=V;
    for j=1:num
        C=XX(:,1+(num-1)*PatchSize:num*PatchSize);
        C=bsxfun(@minus,C,mean_A);
        a=Fnorm(C*V);
        b=Fnorm(C-C*V*V');
        d=1/a/b;
        G=G+d*C'*C;
    end
    H=G*V;
    [A,U,W] = svd(H);
    V=A*I*(W)';
    
    if Fnorm(V-VS)/Fnorm(V)<0.001
        es=0;
    end
end
end

