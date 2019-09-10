%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
% Author: Xi Peng@milab.org, Sichuan University.
% pangsaai@gmail.com
% Date: 6, Seq. 2013
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
addpath('../data/');

addpath('E:\Documents\Experiments\Databases\facial data set\LFW\LFW6\');

% CurData = 'LFW11_FFT';
% CurData = 'LFW11_Gabor';
% CurData = 'LFW11_Gray';
CurData = 'LFW11_LBP';

load (CurData);

% -------------------------------------------------------------------------
% parameter setting
par.nClass        =   length(unique(trainlabels));                 % the number of classes in the subset of AR database
par.nDim          =   size(NewTrain_DAT,1);                 % the eigenfaces dimension
% par.disMetric       =   'cityblock'; % reproduce result: par.gamma=0.7,par.lambda=0.01,par.k=6 with LFW11_FFT
% par.disMetric     =   'seuclidean'; % reproduce result: par.gamma=0.9,par.lambda=0.01,par.k=6 with LFW11_Gabor
% par.disMetric     =   'minkowski'; % reproduce result: par.gamma=0.5,par.lambda=0.01,par.k=5 with LFW11_Gray
% par.disMetric     =   'cosine'; % reproduce result: par.gamma=0.5,par.lambda=0.01,par.k=6 with LFW11_LBP
par.disMetric     =   'spearman';% reproduce result: par.gamma=0.5,par.lambda=0.01,par.k=5 with LFW11_LBP

par.gamma             =   [0.5];             % l2 regularized parameter value
par.k                 =   [5];               % the num of nearest neighbor
par.lambda            =   [0.01];           % penalty parameters;
% par.lambda            =   [1e-4 1e-2 0.1:0.2:1];  % sparsity
% par.k                 =   [1 2 3 4 5 6];          % the num of nearest neighbor
% par.gamma             =   [1e-4 1e-2 0.1:0.2:1];  % locality


par.TraNumOfEachGroup = ceil(size(NewTrain_DAT, 2)/par.nClass);

if par.nDim < size(NewTrain_DAT,1)
    par.FeatureSelType = uint8(1);
else
    par.FeatureSelType = uint8(0);
end;

tr_dat   =   double(NewTrain_DAT);
trls     =   double(trainlabels);
tt_dat   =   double(NewTest_DAT);
ttls     =   double(testlabels);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels

%--------------------------------------------------------------------------
 
% tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat, 1),1]) );
% tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tt_dat, 1),1]) );

Rec=cell(length(par.k),1);


pos = 0;
ID = zeros(1, size(tt_dat, 2));

%----------------------------------------------------------------------
% find k neighbor
knnidx = knnsearch(tr_dat',tt_dat','K',max(par.k), 'Distance', par.disMetric);
clear Tr_DAT Tt_DAT;

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

save (['LCCR_Result_' CurData '_nDim' num2str(par.nDim) '_' par.disMetric]);
