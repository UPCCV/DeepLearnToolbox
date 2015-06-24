function net = cnnsetup(net, x, y)
    %assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' myOctaveVersion]);
    inputmaps = 1;
% B=squeeze(A) 返回和矩阵A相同元素但所有单一维都移除的矩阵B，单一维是满足size(A,dim)=1的维。  
% train_x中图像的存放方式是三维的reshape(train_x',28,28,60000)，前面两维表示图像的行与列，  
% 第三维就表示有多少个图像。这样squeeze(x(:, :, 1))就相当于取第一个图像样本后，再把第三维  
% 移除，就变成了28x28的矩阵，也就是得到一幅图像，再size一下就得到了训练样本图像的行数与列数了
    mapsize = size(squeeze(x(:, :, 1)));%输入层大小,28*28
% 下面通过传入net这个结构体来逐层构建CNN网络  
% n = numel(A)返回数组A中元素个数  
% net.layers中有五个struct类型的元素，实际上就表示CNN共有五层，这里范围的是5  
    for l = 1 : numel(net.layers)   %  layer,numel(net.layers)  表示有多少层  
        if strcmp(net.layers{l}.type, 's')% 如果这层是 子采样层  
            % subsampling层的mapsize，最开始mapsize是每张图的大小28*28  
            % 这里除以scale=2，就是pooling之后图的大小，pooling域之间没有重叠，所以pooling后的图像为14*14  
            % 注意这里的右边的mapsize保存的都是上一层每张特征map的大小，它会随着循环进行不断更新 
            mapsize = mapsize / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps % inputmap就是上一层有多少张特征图  
                net.layers{l}.b{j} = 0;% 将偏置初始化为0  
            end
        end
        if strcmp(net.layers{l}.type, 'c')% 如果这层是 卷积层 
            % 旧的mapsize保存的是上一层的特征map的大小，那么如果卷积核的移动步长是1，那用  
            % kernelsize*kernelsize大小的卷积核卷积上一层的特征map后，得到的新的map的大小就是下面这样  
            mapsize = mapsize - net.layers{l}.kernelsize + 1;%卷积层大小为上层大小-核大小+1,这里的mapsize可以参见UFLDL里面的那张图下面的解释 
            % 该层需要学习的参数个数。每张特征map是一个(后层特征图数量)*(用来卷积的patch图的大小)  
            % 因为是通过用一个核窗口在上一个特征map层中移动（核窗口每次移动1个像素），遍历上一个特征map  
            % 层的每个神经元。核窗口由kernelsize*kernelsize个元素组成，每个元素是一个独立的权值，所以  
            % 就有kernelsize*kernelsize个需要学习的权值，再加一个偏置值。另外，由于是权值共享，也就是  
            % 说同一个特征map层是用同一个具有相同权值元素的kernelsize*kernelsize的核窗口去感受输入上一  
            % 个特征map层的每个神经元得到的，所以同一个特征map，它的权值是一样的，共享的，权值只取决于  
            % 核窗口。然后，不同的特征map提取输入上一个特征map层不同的特征，所以采用的核窗口不一样，也  
            % 就是权值不一样，所以outputmaps个特征map就有（kernelsize*kernelsize+1）* outputmaps那么多的权值了  
            % 但这里fan_out只保存卷积核的权值W，偏置b在下面独立保存  
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;%卷积核初始化，1层卷积为1×6个卷积核，2层卷积一共6×12=72个参数
            for j = 1 : net.layers{l}.outputmaps  %  output map
                % fan_out保存的是对于上一层的一张特征map，我在这一层需要对这一张特征map提取outputmaps种特征，  
                % 提取每种特征用到的卷积核不同，所以fan_out保存的是这一层输出新的特征需要学习的参数个数  
                % 而，fan_in保存的是，我在这一层，要连接到上一层中所有的特征map，然后用fan_out保存的提取特征  
                % 的权值来提取他们的特征。也即是对于每一个当前层特征图，有多少个参数链到前层
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;%//隐藏层的大小，是一个(后层特征图数量)*(用来卷积的patch图的大小)  
                for i = 1 : inputmaps  %  input map,对于每一个后层特征图，有多少个参数链到前层  
                     % 随机初始化权值，也就是共有outputmaps个卷积核，对上层的每个特征map，都需要用这么多个卷积核  
                    % 去卷积提取特征。  
                    % rand(n)是产生n×n的 0-1之间均匀取值的数值的矩阵，再减去0.5就相当于产生-0.5到0.5之间的随机数  
                    % 再 *2 就放大到 [-1, 1]。然后再乘以后面那一数，why？  
                    % 反正就是将卷积核每个元素初始化为[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]  
                    % 之间的随机数。因为这里是权值共享的，也就是对于一张特征map，所有感受野位置的卷积核都是一样的  
                    % 所以只需要保存的是 inputmaps * outputmaps 个卷积核。 
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));%设置每层的权重，权重设置为：-1~1之间的随机数/sqrt（6/（输入神经元数量+输出神经元数量））
                end
                net.layers{l}.b{j} = 0;% % 将偏置初始化为0  
            end
            % 只有在卷积层的时候才会改变特征map的个数，pooling的时候不会改变个数。这层输出的特征map个数就是  
            % 输入到下一层的特征map个数
            inputmaps = net.layers{l}.outputmaps;%把上一层的输出变成下一层的输入
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
     % fvnum 是输出层的前面一层的神经元个数。  
    % 这一层的上一层是经过pooling后的层，包含有inputmaps个特征map。每个特征map的大小是mapsize。  
    % 所以，该层的神经元个数是 inputmaps * （每个特征map的大小）  
    % prod: Product of elements.  
    % For vectors, prod(X) is the product of the elements of X  
    % 在这里 mapsize = [特征map的行数 特征map的列数]，所以prod后就是 特征map的行*列 
    fvnum = prod(mapsize) * inputmaps;%prod为矩阵个元素相乘,因此这儿的作用就是计算输出层之前那层神经元的个数，fvnum=4×4×12=192
     % onum 是标签的个数，也就是输出层神经元的个数。你要分多少个类，自然就有多少个输出神经元 
    onum = size(y, 1);%输出层的神经元个数
% 这里是最后一层神经网络的设定  
    net.ffb = zeros(onum, 1);%输出层偏置,这里是最后一层神经网络的设定% ffb 是输出层每个神经元对应的基biases  
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));%输出层权重 % ffW 输出层前一层 与 输出层 连接的权值，这两层之间是全连接的
end
