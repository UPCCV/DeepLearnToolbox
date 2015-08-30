
%在这里根据需要更改文件位置
train_x=loadMNISTImages('../../data/train-images.idx3-ubyte');
train_y=loadMNISTLabels('../../data/train-labels.idx1-ubyte');
test_x=loadMNISTImages('../../data/t10k-images.idx3-ubyte');
test_y=loadMNISTLabels('../../data/t10k-labels.idx1-ubyte');

error = zeros(10,1);

for j=1:10
    tem_test = knnclassify(test_x, train_x, train_y, j, 'euclidean', 'nearest'); %调用KNN方法，返回测试输出
%     tem_test=zeros(10000);
    testLength = length(test_y);
    error(j)=0;
for i=1:testLength
    if(tem_test(i)~=test_y(i))
        error(j)=error(j)+1;%计算误差
    end
end

end


errorRate = error / testLength;
