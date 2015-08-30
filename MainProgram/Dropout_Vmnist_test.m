
errors=zeros(5,1);
% nps=1;
% nps=5;  %设置训练次数
% nps=10;
nps=20;
%************************%
%第一个是basic
%************************%
% 在这里根据需要更改文件位置
disp('basic');
errors(1)=test3('../data/variation_mnist/mnist_train.amat','../data/variation_mnist/mnist_test.amat',nps);



%************************%
%第2个是background
%************************%
% //normalize
disp('background');
errors(2)=test3('..\data\variation_mnist\mnist_background_images_train.amat','..\data\variation_mnist\mnist_background_images_test.amat',nps);


%************************%
%第3个是random background
%************************%
% 在这里根据需要更改文件位置
% //normalize
disp('random background');
errors(3)=test3('..\data\variation_mnist\mnist_background_random_train.amat','..\data\variation_mnist\mnist_background_random_test.amat',nps);



%************************%
%第4个是rotated
%************************%
% 在这里根据需要更改文件位置

disp('rotated');
errors(4)=test3('..\data\variation_mnist\mnist_all_rotation_normalized_float_train_valid.amat','..\data\variation_mnist\mnist_all_rotation_normalized_float_test.amat',nps);


%************************%
%第5个是rotated+background
%************************%
% 在这里根据需要更改文件位置

disp('rotated+background');
errors(5)=test3('..\data\variation_mnist\mnist_all_background_images_rotation_normalized_train_valid.amat','..\data\variation_mnist\mnist_all_background_images_rotation_normalized_test.amat',nps);



