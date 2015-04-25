function net = cnnsetup(net, x, y)
    %assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));
%尤其注意这几个循环的参数的设定 
    for l = 1 : numel(net.layers)   %  layer,numel(net.layers)  表示有多少层  
        if strcmp(net.layers{l}.type, 's')%降采样层
            mapsize = mapsize / net.layers{l}.scale;%subsampling层的mapsize，最开始mapsize是每张图的大小28*28(这是第一次卷积后的结果，卷积前是32*32) 
            %这里除以scale，就是pooling之后图的大小，这里为14*14 
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;%bias统一设置为0
            end
        end
        if strcmp(net.layers{l}.type, 'c')%卷积层
            mapsize = mapsize - net.layers{l}.kernelsize + 1;%卷积层大小为上层大小奥-核大小+1,这里的mapsize可以参见UFLDL里面的那张图下面的解释 
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;%卷积核初始化，1层卷积为1×6个卷积核，2层卷积一共6×12=72个参数
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;%//隐藏层的大小，是一个(后层特征图数量)*(用来卷积的patch图的大小)  
                for i = 1 : inputmaps  %  input map,对于每一个后层特征图，有多少个参数链到前层  
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));%设置每层的权重，权重设置为：-1~1之间的随机数/sqrt（6/（输入神经元数量+输出神经元数量））
                end
                net.layers{l}.b{j} = 0;%设置每层的偏置
            end
            inputmaps = net.layers{l}.outputmaps;%把上一层的输出变成下一层的输入
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;%prod为矩阵个元素相乘,因此这儿的作用就是计算输出层之前那层神经元的个数，fvnum=4×4×12=192
    onum = size(y, 1);%输出层的神经元个数

    net.ffb = zeros(onum, 1);%输出层偏置,这里是最后一层神经网络的设定
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));%输出层权重
end
