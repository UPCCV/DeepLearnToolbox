function [er, bad] = cnntest(net, x, y)
    %输出错误率，错误的索引；输入 训练好的网络，测试样本，测试样本的标签
    %  feedforward
    net = cnnff(net, x);%前向传播
    [~, h] = max(net.o);%找到输出的最大值
    [~, a] = max(y);%找到真实的标签
    bad = find(h ~= a);%找到标签不等的索引

    er = numel(bad) / size(y, 2);%计算错误率。其中y的第二维是测试样本的数量
end
