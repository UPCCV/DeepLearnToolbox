MainProgram 文件夹内的程序功能一览

KNN 文件夹：利用KNN测试的程序都在该文件夹，内部有README.txt能够进行了解

main1.m  ： 运行test1的matlab脚本，读取MNIST数据，训练CNN后进行测试，会展示Error Rate

main2.m  : 运行test2的matlab脚本，首先得设置nps即训练次数， 然后调用test2进入下一步

test2( fileName1, fileName2 , nps ):  三个参数第一个fileName1，需要读取的图像存储文件名，第二个fileName2需要读取的labels， 第三个nps通过nps对训练次数进行设置
	   内部调用loadMNISTImages

loadMNISTImages.m ：通过filename定位文件，打开文件进行读取。id决定读取格式，id=1：读取MNIST文件；id=2，读取变形的MNIST


Dropout_Mnist.m : 内部读取mnist数据，调用NN，从而查看有无dropout，对辨别mnist数据的影响

Dropout_Vmnist.m : 内部调用test3进行读文件，测试NN

test3.m ( fileName1, fileName2 , nps )： fileName1是训练文件名，fileName2是测试文件名，nps是训练次数，这里需要注意variation mnist的数字图像和label都存在一个文件内