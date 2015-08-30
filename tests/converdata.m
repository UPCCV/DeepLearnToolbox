datadir='chardata';
subdir=dir(datadir);
chardata=[];
label=[];
for i=1:length(subdir)
    if( isequal( subdir( i ).name, '.' ) || ...
        isequal( subdir( i ).name, '..' ) || ...
        ~subdir( i ).isdir )   % 如果不是目录跳过
        continue;
    end
    subdirpath = fullfile( datadir, subdir( i ).name, '*.png' );
    images = dir( subdirpath );
    for j=1:length(images)
        imagepath = fullfile( datadir, subdir( i ).name, images( j ).name  );
        imagedata=imread(imagepath);
        %imshow(imagedata);
        imagedata=imresize(imagedata,[28,28]);
        imagedata=reshape(imagedata,[1,784]);
        chardata=[chardata;imagedata];
        onelabel=[];
        for k =1:length(subdir)-3
            dirnum=str2num(subdir(i).name);
            if(dirnum==k-1)
                onelabel=[onelabel,1];
            else
                onelabel=[onelabel,0]; 
            end
        end
        label=[label;onelabel];
    end
end
save 'chardata/chardata.mat' chardata label
