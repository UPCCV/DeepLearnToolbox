function test_example_CNN
%用mnist数据库测试卷积神经网络性能
isOctave=0;
addpath(genpath('../data/'));
addpath(genpath('../CNN/'));
addpath(genpath('../Util/'));
load mnist_uint8;%加载数据

train_x = double(reshape(train_x',28,28,60000))/255;%还原图像数据,并归一化
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};%设置CNN的结构，输入层，卷积层，降采样，卷积，降采样


opts.alpha = 1;%
opts.batchsize = 50;%每批训练数据的大小
opts.numepochs = 1;%训练的代数

cnn = cnnsetup(cnn, train_x, train_y);%设置网络，初始化卷积核、偏置,第一个参数为网络的结构，第二个为训练的样本，第三个为训练的标签
cnn = cnntrain(cnn, train_x, train_y, opts);%训练网络，第一个参数为网络的结构，第二个为训练的样本，第三个为训练的标签，第四个为附加选项

[er, bad] = cnntest(cnn, test_x, test_y);%测试网络，第一个参数为网络的结构，第二个为测试的样本，第三个为测试的标签，返回错误率和错误的标签

%plot mean squared error
figure; plot(cnn.rL);%绘制均方误差曲线
disp([num2str(er*100) '% error']); %显示误差
assert(er<0.12, 'Too big error');
