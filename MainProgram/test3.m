function [error] = test3( fileName1, fileName2, nps)
%TEST3 Summary of this function goes here
%   Detailed explanation goes here


[test_x,test_y]=loadMNISTImages(fileName2,2);  
[train_x,train_y]=loadMNISTImages(fileName1,2);

% %% //without dropout
% disp('without dropout');
% % rng(0);
% nn = nnsetup([784 100 10]);% //初步构造了一个输入-隐含-输出层网络，其中包括了
%                            % //权值的初始化，学习率，momentum，激发函数类型，
%                            % //惩罚系数，dropout等
% opts.numepochs =  nps;   %  //Number of full sweeps through data
% opts.batchsize = 100;  %  //Take a mean gradient step over this many samples
% [nn, L] = nntrain(nn, train_x, train_y, opts);
% [er, bad] = nntest(nn, test_x, test_y);
% str = sprintf('testing error rate is: %f',er);
% disp(str)

%% //with dropout
% rng(0);
disp('with dropout');
nn = nnsetup([784 100 10]);
nn.dropoutFraction = 0.5;   %  //Dropout fraction，每一次mini-batch样本输入训练时，随机扔掉50%的隐含层节点
opts.numepochs =  nps;        %  //Number of full sweeps through data
opts.batchsize = 50;       %  //Take a mean gradient step over this many samples
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
str = sprintf('testing error rate is: %f',er);
disp(str)
error=er;
end

