function [ f ] = Fnorm( X )
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
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

