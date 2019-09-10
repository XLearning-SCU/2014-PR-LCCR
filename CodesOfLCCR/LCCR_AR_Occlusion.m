%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%/
% Author: Xi Peng@milab.org, Sichuan University.
% pangsaai@gmail.com
% Date: 15, Aug. 2011
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
%data loading (here we use the AR dataset as an example)
addpath('.\AR_DAT_RandomOcclude\');
% addpath('.\AR_DAT_RandomOcclude\');

CurData = 'AR_DAT_Disguise_Glass_60_43';
% CurData = 'AR_database_60_43_Occlusion_10';
load (CurData);

% -------------------------------------------------------------------------
% parameter setting
par.nClass        =   100;                 % the number of classes in the subset of AR database
par.nDim          =   300;                 % the eigenfaces dimension
par.disMetric     =   'cityblock'; 
% par.disMetric     =   'seuclidean'; 
% par.disMetric     =   'minkowski'; 
% par.disMetric     =   'cosine'; 
% par.disMetric     =   'spearman';

par.AdditiveColumn = 0; % if D = [A I], where I is a unit matrix, I = eye(size(D, 1));

% par.lambda            =   [0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.2];             % sparsity
% par.k                 =   [1 2 3 4];  % the num of nearest neighbor
% par.gamma             =   [0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.2];  % locality
par.gamma              =   [0.01];             % l2 regularized parameter value
par.k                  =   [1];  % the num of nearest neighbor
par.lambda             =   [0.9];  % penalty parameters;


par.TraNumOfEachGroup = ceil(size(NewTrain_DAT, 2)/par.nClass);

if par.nDim < size(NewTrain_DAT,1)
    par.FeatureSelType = uint8(1);
else
    par.FeatureSelType = uint8(0);
end;

Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
trls     =   trainlabels(trainlabels<=par.nClass);
Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
ttls     =   testlabels(testlabels<=par.nClass);
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

Rec = cell(size(par.k,2), size(par.kappa,2), size(par.alpha,2));
Tcost = zeros(1, size(par.k,2)*size(par.kappa,2)*size(par.alpha,2));
pos = 0;
ID = zeros(1, size(tt_dat, 2));

%----------------------------------------------------------------------
% find k neighbor
knnidx=[];
knnidx = knnsearch(Tr_DAT',Tt_DAT','K',max(par.k), 'Distance', par.disMetric);
tmpClass = par.nClass;
tmpTraNumOfEachGroup = par.TraNumOfEachGroup;
if par.AdditiveColumn == 1
    ExD = [tr_dat eye(size(tr_dat, 1))];
else
    ExD = tr_dat;
end;

for i = 1:size(par.k,2)
%projection matrix computing
    for m = 1:par.k(i)
        if m == 1
            SecTerm = tr_dat(:,knnidx(:,m));
        else
            SecTerm = SecTerm + tr_dat(:,knnidx(:,m));
        end
    end;
    SecTerm = SecTerm./par.k(i);
    clear m;
    %----------------------------------------------------------------------
    for j = 1:size(par.kappa,2)
    %------------------------------------------------------------------
    %computing the project matrix
        
        Proj_M = pinv( ExD'*ExD+ par.kappa(j) * eye( size (ExD,2) ) ) * ExD';
        
        for k = 1:size(par.alpha,2)
            tic;
            coef   =  Proj_M * ( (1 - par.alpha(k)) * tt_dat + par.alpha(k) * SecTerm);
            %--------------------------------------------------------------
     %inference via sparse coding classifier with local information embbeding          
            
            tic;
                for indTest = 1:size(tt_dat,2)
                    ID(indTest) = IDcheck_norm(tr_dat, coef(:,indTest), tt_dat(:,indTest), trls);
                end
%             end            
            cornum      =   sum(ID==ttls);
            % recognition rate  
            Rec{i}{j}{k}   =   [cornum/length(ttls)];             
            pos = pos + 1;
            Tcost(pos) = toc;
            disp(['***The time cost for ' num2str(pos) 'th computation is ' num2str(Tcost(pos))  'seconds']);
            fprintf(['recogniton rate is ' num2str(Rec{i}{j}{k}) '\n']);
        end;
    end;
    
end;
clear ExD pos tmpClass tmpTraNumOfEachGroup SecTerm coef j k Mean_Image Tr_DAT Tt_DAT cornum disc_set disc_value i id indTest tr_dat tt_dat trls Proj_M CurDataPath;
if par.FeatureSelType == 0
    save (['CRC_LIE_Result_RandomSample_' CurData '_' num2str(par.nDim)  '_' par.disMetric]);
else
    save (['CRC_LIE_Result_Eigenface_' CurData '_' num2str(par.nDim) '_' par.disMetric]);
end;
