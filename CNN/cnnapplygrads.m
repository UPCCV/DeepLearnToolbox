function net = cnnapplygrads(net, opts)
    for l = 2 : numel(net.layers)%从第二层开始
        if strcmp(net.layers{l}.type, 'c')%对于每个卷积层，
            for j = 1 : numel(net.layers{l}.a)%枚举改层的每个输出%枚举所有卷积核的net.layers{l}.k{ii}{j}
                for ii = 1 : numel(net.layers{l - 1}.a)%枚举上层的每个输出
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                end
                %修改偏置
                net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
            end
        end
    end
    %单层感知器的更新
    net.ffW = net.ffW - opts.alpha * net.dffW;
    net.ffb = net.ffb - opts.alpha * net.dffb;
end
