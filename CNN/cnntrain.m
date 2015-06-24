function net = cnntrain(net, x, y, opts)
%net为网络，x为训练数据，y为标签，opts为训练参数
    m = size(x, 3);%m为样本数量，size（x）=[28*28*60000]% m 保存的是 训练样本个数  
    numbatches = m / opts.batchsize;%训练时一批数据包含的图片数量
    % rem: Remainder after division. rem(x,y) is x - n.*y 相当于求余  
    % rem(numbatches, 1) 就相当于取其小数部分，如果为0，就是整数  
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];%rL是最小均方差的平滑序列，绘图时使用
    for i = 1 : opts.numepochs%训练迭代，此处为50
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);%显示当前迭代的次数
        tic;%计时开始
        % P = randperm(N) 返回[1, N]之间所有整数的一个随机的序列，例如randperm(6) 可能会返回 [2 4 5 6 1 3]  
        % 这样就相当于把原来的样本排列打乱，再挑出一些样本来训练 
        kk = randperm(m);%打乱样本的顺序
        for l = 1 : numbatches%分成numbatches批，MNIST分了50批，训练每个batch
            % 取出打乱顺序后的batchsize个样本和对应的标签  
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));%获取每批的训练样本和标签
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            % 在当前的网络权值和网络输入下计算网络的输出
            net = cnnff(net, batch_x);%完成前向过程
            % 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值  
            %（也就是那些卷积核的元素）的导数  
            net = cnnbp(net, batch_y);%完成误差传导和梯度计算过程
            % 得到误差对权值的导数后，就通过权值更新方法去更新权值 
            net = cnnapplygrads(net, opts);%应用梯度，模型更新
            if isempty(net.rL)%net.L为模型的costfunction，即最小均方误差，net.rL是平滑后的序列
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;% 保存历史的误差值，以便画图分析  
        end
        toc;%计时结束
    end
    
end
