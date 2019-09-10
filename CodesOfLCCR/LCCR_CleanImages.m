%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
% Author: Xi Peng@milab.org, Sichuan University.
% pangsaai@gmail.com
% Date: 15, Agu. 2011
% Description:  This code is developed for Learning Locality-Constrained Collaborative Representation for Face Recognition [1]. 

% IF your used any part of this code, PLEASE approximately cited our works;
% In addition, we adopted some codes from other works, Please approximately
%              cited the works if you used the corresponding codes or databases.

% Reference:
% [1] Xi Peng, Lei Zhang, Zhang Yi, K. K. Tan.
%     Learning Locality-Constrained Collaborative Representation for Robust Face Recognition
%     arXiv:1210.1316.

% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
% EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL HOLDER AND CONTRIBUTORS BE LIABLE FOR ANY
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/

close all;
clear all;
clc;

%% --------------------------------------------------------------------------
% data loading (here we use the AR dataset as an example)
addpath('..\data\')
CurData = 'AR_database_normal_image_60_43';
load (CurData);

% -------------------------------------------------------------------------
% parameter setting
par.nClass        =   max(trainlabels);                 % the number of classes in the subset of AR database
par.nDim          =   300;                 % the eigenfaces dimension
par.disMetric     =   'cityblock'; 
% par.disMetric     =   'seuclidean'; 
% par.disMetric     =   'minkowski'; 
% par.disMetric     =   'cosine'; 
% par.disMetric     =   'spearman';

% par.lambda            =   [0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.2];             % sparsity
% par.k                 =   [1 2 3 4];  % the num of nearest neighbor
% par.gamma             =   [0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.2];  % locality
par.gamma             =   [0.001];             % l2 regularized parameter value
par.k                 =   [3];  % the num of nearest neighbor
par.lambda             =   [0.3];  % penalty parameters;



par.TraNumOfEachGroup = ceil(size(NewTrain_DAT, 2)/par.nClass);

if par.nDim < size(NewTrain_DAT,1)
    par.FeatureSelType = uint8(1);
else
    par.FeatureSelType = uint8(0);
end;

Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
trls     =   double(trainlabels);
Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
ttls     =   double(testlabels);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels

%--------------------------------------------------------------------------
% eigenface extracting
if par.FeatureSelType == 1
    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,par.nDim);
    tr_dat  =  disc_set'*Tr_DAT;
    tt_dat  =  disc_set'*Tt_DAT;
else
    tr_dat = Tr_DAT;
    tt_dat = Tt_DAT;
end;
tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat, 1),1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tr_dat, 1),1]) );
clear disc_set disc_value Mean_Image;

Rec=cell(length(par.k),1);
pos = 0;
ID = zeros(1, size(tt_dat, 2));

%----------------------------------------------------------------------
% find k neighbor
knnidx = knnsearch(Tr_DAT',Tt_DAT','K',max(par.k), 'Distance', par.disMetric);

for i = 1:size(par.k,2)    
    for m = 1:par.k(i)
        if m == 1
            Neigh_Coef = tr_dat(:,knnidx(:,m));
        else
            Neigh_Coef = Neigh_Coef + tr_dat(:,knnidx(:,m));
        end
    end;
    Neigh_Coef = Neigh_Coef./par.k(i);
    clear m;
    tmpRec = zeros(length(par.lambda),length(par.gamma)); 
    for j = 1:size(par.lambda,2)
    %computing the project matrix  
    Proj_M = pinv ( tr_dat'*tr_dat + par.lambda(j) )*tr_dat';
        for k = 1:size(par.gamma,2)            
            coef = Proj_M*((1-par.gamma(k)).*tt_dat + par.gamma(k).*Neigh_Coef);     
         %inference via sparse coding classifier with local information embbeding 
            for indTest = 1:size(tt_dat,2)
                ID(indTest) = IDcheck_norm(tr_dat, coef(:,indTest), tt_dat(:,indTest), trls);
            end       
            cornum      =   sum(ID==ttls);
            % recognition rate  
            tmpRec(j,k) = [cornum/length(ttls)];
%             fprintf(['recogniton rate is ' num2str(Rec{i}{j}{k})]);
            pos = pos + 1;
            fprintf([num2str(pos) '-th result is ' num2str(tmpRec(j,k)) '\n']);
        end;
    end;
    Rec{i} = tmpRec;
end;
clear Neigh_Coef pos tmpClass tmpTraNumOfEachGroup SecTerm coef j k Mean_Image Tr_DAT Tt_DAT cornum disc_set disc_value i id indTest tr_dat tt_dat trls Proj_M CurDataPath;
clear tmpRec ID knnidx ;
if par.FeatureSelType == 0
    save (['LCCR2_Result_RandomSample_' CurData '_' num2str(par.nDim)  '_' par.disMetric]);
else
    save (['LCCR2_Result_Eigenface_' CurData '_' num2str(par.nDim) '_' par.disMetric]);
end;