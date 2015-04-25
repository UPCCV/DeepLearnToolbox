function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end
%m是训练样本的数量,注意在调用的时候我们设置了opt，batchsize是做batch gradient时候的大小  
m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;%计算batch的数量

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs%迭代的代数
    tic;
    
    kk = randperm(m);%打乱样本顺序,randperm(m)生成一个乱序的1到m的数组  
    for l = 1 : numbatches%每批样本进行处理
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);%
        
        %Add noise to input (for use in denoising autoencoder)加入noise，这是denoising autoencoder需要使用到的部分 这部分请参见《Extracting and Composing Robust Features with Denoising Autoencoders》这篇论文  
        %具体加入的方法就是把训练样例中的一些数据调整变为0，inputZeroMaskedFraction表示了调整的比例  
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);%前向传播
        nn = nnbp(nn);%反向传播及计算梯度
        nn = nnapplygrads(nn);%梯度下降,更新模型
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    t = toc;

    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end
end

