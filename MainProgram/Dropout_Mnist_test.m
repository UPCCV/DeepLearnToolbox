%% //导入minst数据并归一化
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);
% //normalize
[train_x, mu, sigma] = zscore(train_x);% //归一化train_x,其中mu是个行向量,mu是个列向量
test_x = normalize(test_x, mu, sigma);% //在线测试时，归一化用的是训练样本的均值和方差，需要特别注意

%% //without dropout
% rng(0);
nn = nnsetup([784 100 10]);% //初步构造了一个输入-隐含-输出层网络，其中包括了
                           % //权值的初始化，学习率，momentum，激发函数类型，
                           % //惩罚系数，dropout等
opts.numepochs =  5;   %  //Number of full sweeps through data
opts.batchsize = 100;  %  //Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
str = sprintf('testing error rate is: %f',er);
disp(str)

%% //with dropout
% rng(0);
nn = nnsetup([784 100 10]);
nn.dropoutFraction = 0.5;   %  //Dropout fraction，每一次mini-batch样本输入训练时，随机扔掉50%的隐含层节点
opts.numepochs =  5;        %  //Number of full sweeps through data
opts.batchsize = 100;       %  //Take a mean gradient step over this many samples
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
str = sprintf('testing error rate is: %f',er);
disp(str)