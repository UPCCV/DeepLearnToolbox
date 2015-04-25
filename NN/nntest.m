function [er, bad] = nntest(nn, x, y)
%调用一下nnpredict，在和test的集合进行比较
    labels = nnpredict(nn, x);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
