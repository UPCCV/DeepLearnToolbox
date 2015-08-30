load ('chardata/chardata.mat');

num4train=40;%50个样本中有40个来训练

train_x=[];
train_y=[];
test_x=[];
test_y=[];

for i=1:size(chardata)/50
    train_x=[train_x;chardata(1+50*(i-1):num4train+50*(i-1),:)];
    train_y=[train_y;label(1+50*(i-1):num4train+50*(i-1),:)];
    test_x=[test_x;chardata(1+num4train+50*(i-1):50*i,:)];
    test_y=[test_y;label(1+num4train+50*(i-1):50*i,:)];
end