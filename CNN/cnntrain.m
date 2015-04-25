function net = cnntrain(net, x, y, opts)
%net为网络，x为训练数据，y为标签，opts为训练参数
    m = size(x, 3);%m为样本数量，size（x）=[28*28*60000]
    numbatches = m / opts.batchsize;%训练时一批数据包含的图片数量
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];%rL是最小均方差的平滑序列，绘图时使用
    for i = 1 : opts.numepochs%训练迭代，此处为50
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);%显示当前迭代的次数
        tic;%计开始
        kk = randperm(m);%打乱样本的顺序
        for l = 1 : numbatches%分成numbatches批，MNIST分了50批，训练每个batch
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));%获取每批的训练样本和标签
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            net = cnnff(net, batch_x);%完成前向过程
            net = cnnbp(net, batch_y);%完成误差传导和梯度计算过程
            net = cnnapplygrads(net, opts);%应用梯度，模型更新
            if isempty(net.rL)%net.L为模型的costfunction，即最小均方误差，net.rL是平滑后的序列
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;%计时结束
    end
    
end
