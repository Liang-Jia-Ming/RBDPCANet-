function [ f ] = Fnorm( X )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
f=0;
X_row=size(X,1);
X_col=size(X,2);
for i=1:X_row
    for j=1:X_col
        f=f+X(i,j)*X(i,j);
    end
end
f=sqrt(f);


end

