function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);

    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);% rbmup其实就是简单的一句sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');  也就是上面那张图从v到h计算一次，公式是Wx+c
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);%对每一层的rbm进行训练
    end

end
