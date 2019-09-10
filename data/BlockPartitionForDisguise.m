% Version 1.000 
%
% Matlab Code is used for helping in reviewing of our work submitted to IEEE Tran. SMC, titled: ""
% after our work was accepted, we will provide a higher version on our website:
% www.machineilab.org.

% by PENG Xi, E-mail: pangsaai@gmail.com; ZHANG Yi, E-mail: zhangyi@scu.edu.cn, etc.
% Mar. 3. 2012.
% if you have any problem to use this code, please contact us.

close all;
clear all;
clc;

%% --------------------------------------------------------------------------
%data loading (here we use the AR dataset as an example)
DatPath = '.\AR_DAT_RandomOcclude\';
addpath(DatPath);
% addpath('.\AR_DAT_RandomOcclude\');

CurData = 'AR_database_60_43_Occlusion_50';
% CurData = 'AR_DAT_Disguise_Scarve_60_43';
load (CurData);

% -------------------------------------------------------------------------
% parameter setting
par.nClass        =   100;                 % the number of classes in the subset of AR database
par.nDim          =   size(NewTrain_DAT,1);                 % the  dimension

par.BlockPartition = 1;
par.RowSizeOfBlock = [15 15 15 15];
par.ColSizeOfBlock = [21 22];

% following two row just for occluded data
par.CroppedRow = par.RowSize;
par.CroppedCol = par.ColSize;

par.TraNumOfEachGroup = ceil(size(NewTrain_DAT, 2)/par.nClass);

if par.nDim < size(NewTrain_DAT,1)
    par.FeatureSelType = uint8(1);
else
    par.FeatureSelType = uint8(0);
end;

tr_dat   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
trls     =   trainlabels(trainlabels<=par.nClass);
tt_dat   =   double(NewTest_DAT(:,testlabels<=par.nClass));
ttls     =   testlabels(testlabels<=par.nClass);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels

tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat, 1),1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tr_dat, 1),1]) );

BlkTr_dat=cell(size(par.RowSizeOfBlock,2), size(par.ColSizeOfBlock,2));

for i = 1:size(tr_dat, 2)
    tmp = reshape(tr_dat(:,i),par.CroppedRow,[]);
    tmp = mat2cell(tmp, par.RowSizeOfBlock, par.ColSizeOfBlock);
    for j = 1:size(tmp,1)
        for k=1:size(tmp,2)
            BlkTr_dat{j,k}=[ BlkTr_dat{j,k} reshape(tmp{j,k},[],1)];            
        end;
    end;
end;

BlkTt_dat=cell(size(par.RowSizeOfBlock,2), size(par.ColSizeOfBlock,2));

for i = 1:size(tt_dat, 2)
    tmp = reshape(tt_dat(:,i),par.CroppedRow,[]);
    tmp = mat2cell(tmp, par.RowSizeOfBlock, par.ColSizeOfBlock);
    for j = 1:size(tmp,1)
        for k=1:size(tmp,2)
            BlkTt_dat{j,k}=[ BlkTt_dat{j,k} reshape(tmp{j,k},[],1)];            
        end;
    end;
end;
clear i j k tmp tr_dat tt_dat;
save ([DatPath 'BLK_' CurData], 'BlkTr_dat',  'BlkTt_dat',  'trls',  'ttls',  'par');