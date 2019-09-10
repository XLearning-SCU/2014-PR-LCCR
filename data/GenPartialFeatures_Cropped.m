clear;
clc;

load AR_database_60_43;

par.height = 60;
par.width = 43;

NosePar.xRange = [13 28];
NosePar.yRange = [28 41];

LEyePar.xRange = [1 22];
LEyePar.yRange = [15 28];

MouthPar.xRange = [1 par.width-1];
MouthPar.yRange = [41 par.height-1];

NoseTrain_DAT = [];
LEyeTrain_DAT = [];
MouthTrain_DAT = [];

NoseTest_DAT = [];
LEyeTest_DAT = [];
MouthTest_DAT = [];

for i = 1:size(NewTrain_DAT,2)
    tmp1 = reshape(NewTrain_DAT(:,i), par.height, par.width);
    
    tmp2 = tmp1(NosePar.yRange(1):NosePar.yRange(2), NosePar.xRange(1):NosePar.xRange(2));
    NoseTrain_DAT = [NoseTrain_DAT reshape(tmp2,[],1)];
    
    tmp2 = tmp1(LEyePar.yRange(1):LEyePar.yRange(2), LEyePar.xRange(1):LEyePar.xRange(2));
    LEyeTrain_DAT = [LEyeTrain_DAT reshape(tmp2,[],1)];    
 
    tmp2 = tmp1(MouthPar.yRange(1):MouthPar.yRange(2), MouthPar.xRange(1):MouthPar.xRange(2));
    MouthTrain_DAT = [MouthTrain_DAT reshape(tmp2,[],1)];
end;

for i = 1:size(NewTest_DAT,2)
    tmp1 = reshape(NewTest_DAT(:,i), par.height, par.width);
    
    tmp2 = tmp1(NosePar.yRange(1):NosePar.yRange(2), NosePar.xRange(1):NosePar.xRange(2));
    NoseTest_DAT = [NoseTest_DAT reshape(tmp2,[],1)];
    
    tmp2 = tmp1(LEyePar.yRange(1):LEyePar.yRange(2), LEyePar.xRange(1):LEyePar.xRange(2));
    LEyeTest_DAT = [LEyeTest_DAT reshape(tmp2,[],1)];    
 
    tmp2 = tmp1(MouthPar.yRange(1):MouthPar.yRange(2), MouthPar.xRange(1):MouthPar.xRange(2));
    MouthTest_DAT = [MouthTest_DAT reshape(tmp2,[],1)];
end;
clear i tmp1 tmp2 NewTest_DAT NewTrain_DAT;
NewTrain_DAT = NoseTrain_DAT;
NewTest_DAT = NoseTest_DAT;
save (['AR_NOSE_DAT_' num2str(size(NewTrain_DAT, 1))], 'NosePar','NewTrain_DAT','NewTest_DAT','trainlabels','testlabels');
NewTrain_DAT = LEyeTrain_DAT;
NewTest_DAT = LEyeTest_DAT;
save (['AR_LeftEye_DAT_' num2str(size(NewTrain_DAT, 1))], 'LEyePar','NewTrain_DAT','NewTest_DAT','trainlabels','testlabels');
NewTrain_DAT = MouthTrain_DAT;
NewTest_DAT = MouthTest_DAT;
save (['AR_Mouth_DAT_' num2str(size(NewTrain_DAT, 1))],'MouthPar','NewTrain_DAT','NewTest_DAT','trainlabels','testlabels');