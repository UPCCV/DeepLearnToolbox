
%nps=1;%…Ë÷√—µ¡∑¥Œ ˝
% nps=10;
nps=20;

errors=zeros(5,1);

disp('basic');
errors(1)=test2('../data/variation_mnist/mnist_train.amat','../data/variation_mnist/mnist_test.amat',nps);

disp('background');
errors(2)=test2('../data/variation_mnist/mnist_background_images_train.amat','..\data\variation_mnist\mnist_background_images_test.amat',nps);

disp('random background');
errors(3)=test2('../data/variation_mnist/mnist_background_random_train.amat','..\data\variation_mnist\mnist_background_random_test.amat',nps);

disp('rotated');
errors(4)=test2('..\data\variation_mnist\mnist_all_rotation_normalized_float_train_valid.amat','..\data\variation_mnist\mnist_all_rotation_normalized_float_test.amat',nps);

disp('background rotated');
errors(5)=test2('..\data\variation_mnist\mnist_all_background_images_rotation_normalized_train_valid.amat','..\data\variation_mnist\mnist_all_background_images_rotation_normalized_test.amat',nps);

