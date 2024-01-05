clear all; close all; clc; 
addpath('./Utils');                %其他程序所在
addpath('./Liblinear');
make; 

x=[1,2,3;4,5,6;7,8,9];
X=[3,2,1;6,5,4;9,8,7]
f=Fnorm(X)
F=Fnorm(X-x)
Fnorm(X-x)/Fnorm(X)


