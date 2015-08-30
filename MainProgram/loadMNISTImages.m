function [images,labels] = loadMNISTImages(filename,id)
%load train data or test data
%id=1,used in CNN; id=2 used in NN


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

if(strfind(filename,'train'))
    if(id==1)
        images =fscanf(fp,'%f',[785,12000]);
        label = images(785,:);
        images = reshape(images(1:784,:), 28,28, 12000);
        labels = zeros(10,12000);
        for i=1:12000
            labels(label(i)+1,i)=1;
        end
        images = permute(images,[2 1 3]);
    else
         images =fscanf(fp,'%f',[785,12000]);
         label = images(785,:);
         images = images(1:784,:);
          labels = zeros(10,12000);
          for i=1:12000
            labels(label(i)+1,i)=1;
          end
          labels=labels';
          images=images';
    end
    
   
else
    
    if(id==1)
        images =fscanf(fp,'%f',[785,50000]);
        label = images(785,:);
        images = reshape(images(1:784,:), 28,28, 50000);
        labels = zeros(10,50000);
        for i=1:50000
            labels(label(i)+1,i)=1;
        end
        images = permute(images,[2 1 3]);
    else
         images =fscanf(fp,'%f',[785,50000]);
         label = images(785,:);
         images = images(1:784,:);
         labels = zeros(10,50000);
          for i=1:50000
            labels(label(i)+1,i)=1;
          end
           labels=labels';
           images=images';
    end
end


fclose(fp);

% Reshape to #pixels x #examples
% images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
end
