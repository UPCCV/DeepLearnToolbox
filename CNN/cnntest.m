function [er, bad] = cnntest(net, x, y)
    %输出错误率，错误的索引；输入 训练好的网络，测试样本，测试样本的标签
    %  feedforward
    net = cnnff(net, x);% 前向传播得到输出
    [~, h] = max(net.o); % 找到最大的输出对应的标签  
    [~, a] = max(y);% 找到最大的期望输出对应的索引
    bad = find(h ~= a);% 找到他们不相同的个数，也就是错误的次数  

    er = numel(bad) / size(y, 2);%计算错误率。其中y的第二维是测试样本的数量
end
