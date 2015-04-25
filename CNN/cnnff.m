function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;%a是输入map，是一个[28,28,50]的序列
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);%z=zeros([28,28,50]-[4,4,0]=zeros[24,24,50]
                for i = 1 : inputmaps   %  for each input map做卷积，参考UFLDL，这里是对每一个input的特征图做一次卷积，再加起来  
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;%本层的输出作为下层的输入
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps%这里有点绕绕的，它是新建了一个patch来做卷积，但我们要的是pooling，所以它跳着把结果读出来，步长为scale 这里做的是mean-pooling   
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    %尾部单层感知器的处理，收纳到一个vector里面，方便后面用
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)%fv每次拼合如subFeaturemap2[j],包含50个样本
        sa = size(net.layers{n}.a{j});%[4,4,50]
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons，最后一层的perceptrons，数据识别的结果
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));%计算输出

end
