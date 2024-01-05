function [W ] = L1enginvector( XX,PatchSize,NumFilters)
%L1范数提取特征向量
%   此处显示详细说明
N=size(XX,2);
num=N/PatchSize;
W=zeros(PatchSize,NumFilters);
WS=zeros(PatchSize,NumFilters);
M=zeros(PatchSize,NumFilters);
P=zeros(NumFilters,1);
p=zeros(NumFilters,1);
B=zeros(PatchSize,NumFilters);
mini = min(PatchSize, NumFilters); 
A=zeros(PatchSize,PatchSize);
for i=1:mini
    B(i,i)=1;
    W(i,i)=1;
end
for i=1:num
    A=A+XX(:,1+(num-1)*PatchSize:num*PatchSize);
end
mean_A=A/num;
es=1;
while es
    WS=W;
    for j=1:num
        C=XX(:,1+(num-1)*PatchSize:num*PatchSize);
        C=bsxfun(@minus,C,mean_A);
        for k=1:PatchSize
            p=(W)'*(C(1,:))';
            for kk=1:NumFilters
                if p(kk,:)>=0
                    P(kk,:)=1;
                else P(kk,:)=-1;
                end
            end
            M=M+(C(1,:))'*(P)';
        end
    end
    [U,S,V] = svd(M);
    W=U*B*(V)';
    es=isequal(W,WS);
end
end

