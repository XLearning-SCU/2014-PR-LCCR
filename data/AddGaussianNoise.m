function NewTest_DAT = AddGaussianNoise(NewTest_DAT,noiseFactor)
% noiseFactor = [10 20 30 ... 100]

A = zeros(size(NewTest_DAT));
for i=1:size(NewTest_DAT,2)
    %rand('seed',i);
    A(:,i) = uint8(double(NewTest_DAT(:,i))+noiseFactor*randn(size(NewTest_DAT,1),1));
    %         B(:,i) = uint8(double(NewTrain_DAT(:,i))+noiseFactor*randn(size(NewTrain_DAT,1),1));
    %         subplot(2,1,1)
    %         imshow(uint8(reshape(A(:,i),par.OriRowSize,[])))
    %         subplot(2,1,2)
    %         imshow(uint8(reshape(NewTest_DAT(:,i),par.OriRowSize,[])))
    %         pause(1);
end;
NewTest_DAT = A;
% NewTrain_DAT = double(B);
clear i A B isCorruptted;
