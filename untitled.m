fsolver.net.set_input_dim([0, 1, 3, roi_size, roi_size]);

roi1 = ext_roi(im1, location, center_off,  roi_size, roi_scale_factor);
roi2 = ext_roi(im1, location, [-40,40],  roi_size, roi_scale_factor);
roi3 = ext_roi(im1, location, [0,0],  roi_size, roi_scale_factor*0.6);
roi4 = ext_roi(im1, location, [0,0],  roi_size, roi_scale_factor*1.7);
map1 =  GetMap(size(im1), fea_sz, roi_size, location, center_off, roi_scale_factor, map_sigma_factor, 'trans_gaussian');
map3 =  GetMap(size(im1), fea_sz, roi_size, location, center_off, roi_scale_factor * 0.6, map_sigma_factor, 'trans_gaussian');

% figure(6);
% subplot(1,2,1); imshow(map1);
% subplot(1,2,2); imshow(map3);

figure(2)
subplot(1,2,1); imshow(mat2gray(roi1));
figure(3)
subplot(1,2,1); imshow(mat2gray(roi2));
figure(4)
subplot(1,2,1); imshow(mat2gray(roi3));
figure(5)
subplot(1,2,1); imshow(mat2gray(roi4));


roi1 = impreprocess(roi1);
roi2 = impreprocess(roi2);
roi3 = impreprocess(roi3);
roi4 = impreprocess(roi4);

pre1 = fsolver.net.forward({single(roi1)});
pre2 = fsolver.net.forward({single(roi2)});
pre3 = fsolver.net.forward({single(roi3)});
pre4 = fsolver.net.forward({single(roi4)});
pre1 = permute(pre1{1}, [2,1,3]);
pre2 = permute(pre2{1}, [2,1,3]);
pre3 = permute(pre3{1}, [2,1,3]);
pre4 = permute(pre4{1}, [2,1,3]);

figure(2)
subplot(1,2,2); imshow(mat2gray(pre1));
figure(3)
subplot(1,2,2); imshow(mat2gray(pre2));
figure(4)
subplot(1,2,2); imshow(mat2gray(pre3));
figure(5)
subplot(1,2,2); imshow(mat2gray(pre4));

 im2_name = sprintf([data_path 'img/%0' num2str(num_z) 'd.jpg'], 4);
 im2 = double(imread(im2_name));
 if size(im2,3)~=3
        im2(:,:,2) = im2(:,:,1);
        im2(:,:,3) = im2(:,:,1);
 end 
    
fs = -7:7;
for i = 1:length(fs)
%     center_off = rand(1,2) * 200 - 100;
   f = roi_scale_factor * 1.1 ^fs(i);
    roi = ext_roi(im2, location, [0,0],  roi_size, f);
    figure(7)
    subplot(1,2,1); imshow(mat2gray(roi));
    roi = impreprocess(roi);
    pre = fsolver.net.forward({single(roi)});
    pre = permute(pre{1}, [2,1,3]);
    subplot(1,2,2); imshow(pre);
    pause;
end


roi = ext_roi(im1, location, [0,0],  roi_size, roi_scale_factor);
roi = impreprocess(roi);
pre = fsolver.net.forward({single(roi)});
% a = fsolver.net.params('conva_1', 1).get_data();
% for i = 0:length(a(:))/512-1
% figure(22); bar(a(i*512+1:i*512+512)); pause
% end

a = fsolver.net.blobs('conva_1_5').get_data();
b = fsolver.net.blobs('conva_1_5_down').get_data();
for i = 1:size(a, 3)
    figure(22); subplot(1,2,1);imagesc(permute(b(:,:,i), [2,1,3])); title(num2str(i))
figure(22); subplot(1,2,2);imagesc(permute(a(:,:,i), [2,1,3])); title(num2str(i));pause
end