function labels = nnpredict(nn, x)
%predict不过是nnff一次，得到最后的output
    nn.testing = 1;
    nn = nnff(nn, x, zeros(size(x,1), nn.size(end)));
    nn.testing = 0;
    
    [dummy, i] = max(nn.a{end},[],2);
    labels = i;
end
