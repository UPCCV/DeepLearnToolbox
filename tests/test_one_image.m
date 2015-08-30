function test_one_image
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
num2test=1000;
load 'cnnmodel'
image=test_x(:,:,num2test);
imshow(image');
test_one(:,:,1)=image;
test_one(:,:,2)=image;
net = cnnff(cnn, test_one);% 前向传播得到输出
[~, h] = max(net.o); % 找到最大的输出对应的标签 
[~, a] = max(test_y);% 找到最大的期望输出对应的索引
h(1)-1,a(num2test)-1


