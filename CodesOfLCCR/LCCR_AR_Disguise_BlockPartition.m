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
addpath('.\AR_DAT_Disguise\');
% addpath('.\AR_DAT_RandomOcclude\');

CurData = 'BLK_AR_DAT_Disguise_Glass_60_43';
% CurData = 'BLK_AR_database_60_43_Occlusion_10';
load (CurData);


% -----------------------------------------% parameter setting
par.nClass        =   100;                 % the number of classes in the subset of AR database
par.TraNumOfEachGroup = ceil(size(BlkTr_dat{1,1}, 2)/par.nClass);
par.disMetric     =   'cityblock'; 
% par.disMetric     =   'seuclidean'; 
% par.disMetric     =   'minkowski'; 
% par.disMetric     =   'cosine'; 
% par.disMetric     =   'spearman';

% par.lambda            =   [0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.2];             % sparsity
% par.k                 =   [1 2 3 4];  % the num of nearest neighbor
% par.gamma             =   [0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.2];  % locality
par.gamma             =   [5e-5];             % l2 regularized parameter value
par.k                 =   [6];  % the num of nearest neighbor
par.lambda             =   [0.5];  % penalty parameters;


par.isblock = 1;
par.blocknum = (size(BlkTr_dat,1)*size(BlkTr_dat,2));
knnidx = [];
ID = cell(par.blocknum,1);
pos = 0;

for blkId = 1 : par.blocknum
    tr_dat = BlkTr_dat{blkId};
    tt_dat = BlkTt_dat{blkId};   
    
    knnidx = knnsearch(tr_dat',tt_dat','K',max(par.k), 'Distance', par.disMetric);
    disp(['****The ' num2str(blkId) 'th block is processing!****' ]);
    for i = 1:size(par.k,2)
        %projection matrix computing
        for m = 1:par.k(i)
            if m == 1
                SecTerm = tr_dat(:,knnidx(:,m));
            else
                SecTerm = SecTerm + tr_dat(:,knnidx(:,m));
            end
        end;
        SecTerm = SecTerm./size(knnidx,2);
        clear m;
        %----------------------------------------------------------------------
        for j = 1:size(par.kappa,2)
            %------------------------------------------------------------------
            %computing the project matrix
            
            Proj_M = pinv( tr_dat'*tr_dat+ par.kappa(j) * eye( size (tr_dat,2) ) ) * tr_dat';
            
            for k = 1:size(par.alpha,2)                
                coef   =  Proj_M * ( (1 - par.alpha(k)) * tt_dat + par.alpha(k) * SecTerm);
                %--------------------------------------------------------------
                %inference via sparse coding classifier with local information embbeding   
                
                for indTest = 1:size(tt_dat,2)
                    tmp(indTest) = IDcheck_norm(tr_dat, coef(:,indTest), tt_dat(:,indTest), trls);
                end;
                
                ID{blkId} = [ID{blkId} ; tmp];
                pos = pos + 1;
                fprintf(['the ' num2str(pos) 'th computation is finished!\n']);
            end;
        end;
    end;       
end;

clear pos  SecTerm coef j k Mean_Image Tr_DAT Tt_DAT cornum disc_set disc_value i id indTest tr_dat tt_dat trls Proj_M CurDataPath;

% determine the finnal ID based on the IDs of all blocks by voting method
tmpRec = [];
if max(ttls) ~= par.nClass
    fprintf('the labels are not continuous');
else
    for i = 1:size(ID{1},1)
        for j=1:par.blocknum
            tmp(j,:) = ID{j}(i,:);
        end;
        
        for j = 1:par.nClass
            VoteMatrix(j,:) = sum(ones(size(tmp)) * j == tmp);
        end;
        [tmpc FID]=max(VoteMatrix);
        tmpRec = [tmpRec sum(FID==ttls)/length(ttls)];
    end;
end;
clear tmpc FID VoteMatrix i j k blkId tmp BlkTr_dat BlkTt_dat;

Rec = cell(length(par.k), length(par.kappa),length(par.alpha));
pos = 1;
for i = 1:length(par.k)
    for j = 1:length(par.kappa)
        for k=1:length(par.alpha)            
            Rec{i}{j}{k} = tmpRec(pos);
            pos = pos + 1;
        end;
    end;
end;
clear i j k tmpRec pos;

save (['LCCR_Result_RandomSample_' CurData '_' par.disMetric]);


