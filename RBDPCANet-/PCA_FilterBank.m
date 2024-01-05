function V = PCA_FilterBank(InImg, PatchSize, NumFilters) 
% =======INPUT=============
% InImg            Input images (cell structure)  
% PatchSize        the patch size, asumed to an odd number.
% NumFilters       the number of PCA filters in the bank.
% =======OUTPUT============
% V                PCA filter banks, arranged in column-by-column manner
% =========================

addpath('./Utils')

% to efficiently cope with the large training samples, if the number of training we randomly subsample 10000 the
% training set to learn PCA filter banks
ImgZ = length(InImg);
%MaxSamples = 100000;
%NumRSamples = min(ImgZ, MaxSamples); 
%RandIdx = randperm(ImgZ);
%RandIdx = RandIdx(1:NumRSamples);

%% Learning PCA filters (V)
NumChls = size(InImg{1},3);
Rx = zeros(NumChls*PatchSize^2,NumChls*PatchSize^2);
Xx = cell(ImgZ,1);
Xy = cell(ImgZ,1);
for i =1:ImgZ% RandIdx %1:ImgZ
    im = im2col_mean_removal(InImg{i},[PatchSize PatchSize]); % collect all the patches of the ith image in a matrix, and perform patch mean removal
    [Xx_i, Xy_i] = extT(im,PatchSize);
    Xx{i}=Xx_i;
    Xy{i}=Xy_i;
    %Rx = Rx + im*im'; % sum of all the input images' covariance matrix
end
N=size(Xx{1},2);
XX=zeros(PatchSize,(N*ImgZ));
XY=zeros(PatchSize,(N*ImgZ));
for j=1:ImgZ
    XX(:,(j*N-N+1):(j*N))=Xx{j};
    XY(:,(j*N-N+1):(j*N))=Xy{j};
end
[WX ] = Angleenginvector(XX,PatchSize,NumFilters);
[WY ] = Angleenginvector(XY,PatchSize,NumFilters);
W=cell(NumFilters,1);
V=ones(PatchSize*PatchSize,NumFilters);
for i=1:NumFilters
    W{i}=WX(:,i)*(WY(:,i))';
    V(:,i)=reshape(W{i},PatchSize*PatchSize,1);
end



 



