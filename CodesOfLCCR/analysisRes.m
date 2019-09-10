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


% this file is for analyzing the result of classication using LCCR

clc;
disp(' This file just for analyzing the result of CRC_LIE, which is the generalized form of L2 and CRC method ')
fprintf('\n');

disp(['For data set:' CurData])
fprintf('\n');

disp(['KNN metric is:' par.disMetric '; Dimension is: ' num2str(par.nDim)])
fprintf('\n');

k1 = [];
k2 = [];
k3 = [];
k4 = [];
k5 = [];
k6 = [];
for i=1:size(par.kappa,2)
    k1 = [k1 cell2mat(Rec{1}{i})'];
end;
for i=1:size(par.kappa,2)
    k2 = [k2 cell2mat(Rec{2}{i})'];
end;
for i=1:size(par.kappa,2)
    k3 = [k3 cell2mat(Rec{3}{i})'];
end;
for i=1:size(par.kappa,2)
    k4 = [k4 cell2mat(Rec{4}{i})'];
end;
for i=1:size(par.kappa,2)
    k5 = [k5 cell2mat(Rec{5}{i})'];
end;
for i=1:size(par.kappa,2)
    k6 = [k6 cell2mat(Rec{6}{i})'];
end;   

disp(' ----------------------- the result of CRC_LIE ----------------------- ')

CRC_LIE_val{1} = max(max(k1));
[r{1} c{1}]=find(k1==max(max(k1)));
% disp([' == when the number of neighbor is: ', num2str(1)])
% disp([' kappa = ', num2str(par.kappa(c{1})), ' alpha = ', num2str(par.alpha(r{1}))])
% disp([' max recognition rate is: ', num2str(max(max(k1)))]);
% fprintf('\n');


CRC_LIE_val{2} = max(max(k2));
[r{2} c{2}]=find(k2==max(max(k2)));
% disp([' == when the number of neighbor is: ', num2str(2)])
% disp([' kappa = ', num2str(par.kappa(c{2})), ' alpha = ', num2str(par.alpha(r{2}))])
% disp([' max recognition rate is: ', num2str(max(max(k2)))]);
% fprintf('\n');

CRC_LIE_val{3} = max(max(k3));
[r{3} c{3}]=find(k3==max(max(k3)));
% disp([' == when the number of neighbor is: ', num2str(3)])
% disp([' kappa = ', num2str(par.kappa(c{3})), ' alpha = ', num2str(par.alpha(r{3}))])
% disp([' max recognition rate is: ', num2str(max(max(k3)))]);
% fprintf('\n');

CRC_LIE_val{4} = max(max(k4));
[r{4} c{4}]=find(k4==max(max(k4)));
% disp([' == when the number of neighbor is: ', num2str(4)])
% disp([' kappa = ', num2str(par.kappa(c{4})), ' alpha = ', num2str(par.alpha(r{4}))])
% disp([' max recognition rate is: ', num2str(max(max(k4)))]);
% fprintf('\n');

CRC_LIE_val{5} = max(max(k5));
[r{5} c{5}]=find(k5==max(max(k5)));
% disp([' == when the number of neighbor is: ', num2str(5)])
% disp([' kappa = ', num2str(par.kappa(c{5})), ' alpha = ', num2str(par.alpha(r{5}))])
% disp([' max recognition rate is: ', num2str(max(max(k5)))]);
% fprintf('\n');

CRC_LIE_val{6} = max(max(k6));
[r{6} c{6}]=find(k6==max(max(k6)));
% disp([' == when the number of neighbor is: ', num2str(6)])
% disp([' kappa = ', num2str(par.kappa(c{6})), ' alpha = ', num2str(par.alpha(r{6}))])
% disp([' max recognition rate is: ', num2str(max(max(k4)))]);
% fprintf('\n');

[tmpc tmpi] = max(cell2mat(CRC_LIE_val));

disp([' ***************** when the number of neighbor is: ', num2str(tmpi)])
disp([' kappa =  ', num2str(par.kappa(c{tmpi})), ' alpha =  ', num2str(par.alpha(r{tmpi}))])
disp(['global max recognition rate is: ', num2str(max(max(tmpc)))]);
clear tmpi tmpc;
fprintf('\n');

disp(' ----------------------- the result of CRC, alpha is 0 ----------------------- ')
if par.kappa(1) ~= 0
    disp('this data not includes the result of CRC, please run it again! ');
    fprintf('\n');
else
    CRC_val = max (k1(1,:));
    [CRC_r CRC_c] = find(k1(1,:)==max (k1(1,:))); 
     
    disp([' **** when kappa = ', num2str(par.kappa(CRC_c)),...
     ', global max recognition rate of CRC is: ', num2str(max(max(CRC_val)))]);

end;

disp(' ----------------------- the result of CRC, alpha is 0, kappa is 0 ----------------------- ')
if par.kappa(1) ~= 0 || par.alpha(1) ~= 0
    disp('this data not includes the result of L2, please run it again! ');
else
    L2_val = k1(1,1);  
    disp([' **** global max recognition rate of L2 is: ', num2str(L2_val)]);
end;