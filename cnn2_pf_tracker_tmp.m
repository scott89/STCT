function  cnn2_pf_tracker(set_name, im1_id, ch_num)
cleanupObj = onCleanup(@cleanupFun);
set_tracker_param;
% caffe('presolve_gnet');
% caffe('presolve_lnet');
%% read images
im1_name = sprintf([data_path 'img/%04d.jpg'], im1_id);
im1 = double(imread(im1_name));
if size(im1,3)~=3
    im1(:,:,2) = im1(:,:,1);
    im1(:,:,3) = im1(:,:,1);
end

%% extract roi and display
roi1 = ext_roi(im1, location, l1_off,  roi_size, s1);
%% save roi images
%% ------------------------------
figure(1)
imshow(mat2gray(roi1));
%% ==========================
roi1 = imresize(roi1, [280, 280]);
roi1 = impreprocess(roi1);
fsolver.net.set_net_phase('test');
fsolver.net.cnn2fcn();
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');
feature_blob5 = fsolver.net.blobs('conv5_3');
feature_input.set_data(single(roi1));
fsolver.net.forward_prefilled();
% fea = fsolver.net.forward({roi1});
% fea1 = caffe('forward', {single(roi1)});
lfea1 = feature_blob4.get_data();
%% preprocess roi
roi1 = imresize(roi1, [361, 361]);
% roi1 = impreprocess(roi1);
fsolver.net.set_net_phase('test');
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');
feature_blob5 = fsolver.net.blobs('conv5_3');
fsolver.net.fcn2cnn();
fsolver.net.set_input_dim([0, 1, 3, 361, 361]);
feature_input.set_data(single(roi1));
fsolver.net.forward_prefilled();
% fea = fsolver.net.forward({roi1});

% fea1 = caffe('forward', {single(roi1)});
lfea1 = feature_blob4.get_data();
fea_sz = size(lfea1);
% ch_num = size(fea1,3);
% ch_num = 128;
% fea_sz = size(fea1{1});
% lfea1 = fea1{1};
% gfea1 = imresize(fea1{2}, fea_sz(1:2));
gfea1 = imresize(feature_blob5.get_data(), fea_sz(1:2));
%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% max_iter_select = 100;
max_iter = 50;
map1 =  GetMap(size(im1), fea_sz, roi_size, location, l1_off, s1, 'trans_gaussian');


% conf_store = 0;
% %% select
% caffe('set_phase_train');
% for i=1:max_iter_select
%     l_pre_map = caffe('forward_lnet', {lfea1});
%     l_pre_map = l_pre_map{1};
%     figure(1011); subplot(1,2,1); imagesc(permute(l_pre_map,[2,1,3]));
%     l_diff = l_pre_map-permute(single(map1), [2,1,3]);
%     caffe('backward_lnet', {single(l_diff)});
%     caffe('update_lnet');
%     
%     g_pre_map = caffe('forward_gnet', {gfea1});
%     g_pre_map = g_pre_map{1};
%     figure(1011); subplot(1,2,2); imagesc(permute(g_pre_map,[2,1,3]));
%     g_diff = g_pre_map-permute(single(map1), [2,1,3]);
%     caffe('backward_gnet', {single(g_diff)});
%     caffe('update_gnet');
%     fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(l_diff(:))), sum(abs(g_diff(:))));
% end
% 
% 
% [lsal, lid] = compute_saliency({lfea1}, map1, 'lsolver');
% [gsal, gid] = compute_saliency({gfea1}, map1, 'gsolver');
% 
% lid = lid(1:ch_num);
% gid = gid(1:ch_num);
% 
% lfea1 = lfea1(:,:,lid);
% gfea1 = gfea1(:,:,gid);
% fea2_store = lfea1;
% map2_store = map1;
% l_distractor_store = false;
%% train
% caffe('init_lsolver', lnet_solver_def_file, model_file);
% caffe('init_gsolver', gnet_solver_def_file, model_file);
% caffe('set_phase_train');
% caffe('presolve_gnet');
% caffe('presolve_lnet');

cos_win = single(hann(fea_sz(1)) * hann(fea_sz(2))');
lfea1 = bsxfun(@times, lfea1, cos_win);
gfea1 = bsxfun(@times, gfea1, cos_win);
lsolver.net.set_net_phase('train');
gsolver.net.set_net_phase('train');
lnet_input = lsolver.net.blobs('data');
gnet_input = gsolver.net.blobs('data');
lnet_output = lsolver.net.blobs('conv5_f2');
gnet_output = gsolver.net.blobs('conv5_f2');
for i=1:max_iter
    gsolver.net.empty_net_param_diff();
    lsolver.net.empty_net_param_diff();
%     lnet_input.set_data(lfea1);
%     lsolver.net.forward_prefilled();
%     gnet_input.set_data(gfea1);
%     gsolver.net.forward_prefilled();
%     l_pre_map = lnet_output.get_data();
%     g_pre_map = gnet_output.get_data();
    l_pre_map = lsolver.net.forward({lfea1});
    g_pre_map = gsolver.net.forward({gfea1});
    l_pre_map = l_pre_map{1};
    g_pre_map = g_pre_map{1};
    figure(1011); subplot(1,2,1); imagesc(permute(l_pre_map,[2,1,3]));
    figure(1011); subplot(1,2,2); imagesc(permute(g_pre_map,[2,1,3]));
    l_diff = l_pre_map-permute(single(map1), [2,1,3]);
    g_diff = g_pre_map-permute(single(map1), [2,1,3]);
%     lnet_output.set_diff(l_diff);
%     gnet_output.set_diff(g_diff);
%     lsolver.net.backward_prefilled();
%     gsolver.net.backward_prefilled();
    lsolver.net.backward({l_diff});
    gsolver.net.backward({g_diff});
    lsolver.apply_update();
    gsolver.apply_update();
    fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(l_diff(:))), sum(abs(g_diff(:))));
  
    
%     l_pre_map = caffe('forward_lnet', {lfea1});
%     l_pre_map = l_pre_map{1};
%     
%     
%     figure(1011); subplot(1,2,1); imagesc(permute(l_pre_map,[2,1,3]));
%     g_pre_map = caffe('forward_gnet', {gfea1});
%     g_pre_map = g_pre_map{1};
%     
%     figure(1011); subplot(1,2,2); imagesc(permute(g_pre_map,[2,1,3]));
%     
%     l_diff = l_pre_map-permute(single(map1), [2,1,3]);
%     input_diff = caffe('backward_lnet', {single(l_diff)});
%     caffe('update_lnet');
%     g_diff = g_pre_map-permute(single(map1), [2,1,3]);
%     input_diff = caffe('backward_gnet', {single(g_diff)});
%     caffe('update_gnet');
%     fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(l_diff(:))), sum(abs(g_diff(:))));
end
%% ================================================================
%% Init Scale Estimator
scale_param = init_scale_estimator(im1, location, scale_param);


t=0;
fnum = size(GT,1);
position = zeros(6, fnum);
close all


for im2_id = im1_id:fnum
    tic;
    lsolver.net.set_net_phase('test');
    gsolver.net.set_net_phase('test');
    location_last = location;
    tic
    fprintf('Processing Img: %d/%d \n', im2_id, fnum);
    im2_name = sprintf([data_path 'img/%04d.jpg'], im2_id);
    im2 = double(imread(im2_name));
    if size(im2,3)~=3
        im2(:,:,2) = im2(:,:,1);
        im2(:,:,3) = im2(:,:,1);
    end
    
    %% extract roi and display
    [roi2, roi_pos, padded_zero_map, pad] = ext_roi(im2, location, l2_off,  roi_size, s2);
    %% preprocess roi
    roi2 = impreprocess(roi2);
    feature_input.set_data(single(roi2));
    fsolver.net.forward_prefilled();
    
    lfea2 = feature_blob4.get_data();
    gfea2 = imresize(feature_blob5.get_data(), fea_sz(1:2));
    %% compute confidence map
    lfea2 = bsxfun(@times, lfea2, cos_win);
    gfea2 = bsxfun(@times, gfea2, cos_win);
    
    l_pre_map = lsolver.net.forward({lfea2});
    l_pre_map_train = l_pre_map{1};
    l_pre_map = permute(l_pre_map{1}, [2,1,3])/(max(l_pre_map{1}(:))+eps);
    g_pre_map = gsolver.net.forward({gfea2});
    g_pre_map_train = g_pre_map{1};
    g_pre_map = permute(g_pre_map{1}, [2,1,3])/(max(g_pre_map{1}(:))+eps);
    
%     lnet_input.set_data(lfea2);
%     gnet_input.set_data(gfea2);
%     lsolver.net.forward_prefilled();
%     gsolver.net.forward_prefilled();
%     l_pre_map = lnet_output.get_data();
%     g_pre_map = gnet_output.get_data();
    


    
    if im2_id == im1_id
       figure('Number','off', 'Name','Target Heat Maps');
       subplot(1,2,1);
       im_handle1 = imagesc(l_pre_map);
       subplot(1,2,2);
       im_handle2 = imagesc(g_pre_map);
    else
       set(im_handle1, 'CData', l_pre_map)
       set(im_handle2, 'CData', g_pre_map)
    end
    %% compute global confidence
    g_roi_map = imresize(g_pre_map, roi_pos(4:-1:3));
    g_im_map = padded_zero_map;
    g_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = g_roi_map;
    g_im_map = g_im_map(pad+1:end-pad, pad+1:end-pad);
    [g_y, g_x] = find(g_im_map == max(g_im_map(:)), 1);
    
    
    %% compute local confidence
    l_roi_map = imresize(l_pre_map, roi_pos(4:-1:3));
    l_im_map = padded_zero_map;
    l_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = l_roi_map;
    l_im_map = l_im_map(pad+1:end-pad, pad+1:end-pad);
    [l_y, l_x] = find(l_im_map == max(l_im_map(:)), 1);
    if length(max(l_im_map(:))) > 1
        1;
    end
     if length(max(g_im_map(:))) > 1
        1;
    end
    
    %% global scale estimation
    base_location_g = [g_x, g_y, location([3,4])];
%     pos = GT(im2_id, 1:2) + floor(GT(im2_id, 3:4)/2);
%     base_location_g = [pos(1:2), location([3,4])];
    xs = get_scale_sample(im2, base_location_g, scale_param.scaleFactors, scale_param.scale_window, scale_param.scale_model_sz);
    % calculate the correlation response of the scale filter
    xsf = fft(xs,[],2);
    scale_response_g = real(ifft(sum(scale_param.sf_num .* xsf, 1) ./ (scale_param.sf_den + scale_param.lambda)));
    % find the maximum scale response
    max_scale_respones_g = max(scale_response_g(:));
    
    %% local scale estimation
    base_location_l = [l_x, l_y, location([3,4])];
%     base_location_l = [pos(1:2), location([3,4])];
    xs = get_scale_sample(im2, base_location_l, scale_param.scaleFactors, scale_param.scale_window, scale_param.scale_model_sz);
    % calculate the correlation response of the scale filter
    xsf = fft(xs,[],2);
    scale_response_l = real(ifft(sum(scale_param.sf_num .* xsf, 1) ./ (scale_param.sf_den + scale_param.lambda)));
    % find the maximum scale response
    max_scale_respones_l = max(scale_response_l(:));
    
    if max_scale_respones_l >= max_scale_respones_g %|| 1
        recovered_scale = find(scale_response_l == max_scale_respones_l, 1);
        pos = [l_x, l_y];
    else
        recovered_scale = find(scale_response_g == max_scale_respones_g, 1);
        pos = [g_x, g_y];
    end
    
    % update the scale
    scale_param.currentScaleFactor = scale_param.scaleFactors(recovered_scale);
    target_sz = location([3, 4]) * scale_param.currentScaleFactor;
    location = [pos(1) - floor(target_sz(1)/2), pos(2) - floor(target_sz(2)/2), target_sz(1), target_sz(2)];
    t = t + toc;
  
    %% Update lnet and gnet
    if     rand(1)>0.5%mod(im2_id, 1) == 0
        l_off = location_last(1:2)-location(1:2);
        map2 = GetMap(size(im2), fea_sz, roi_size, floor(location), floor(l_off), s2, 'trans_gaussian');
        
        
        lsolver.net.set_net_phase('train');
        gsolver.net.set_net_phase('train');
        gsolver.net.empty_net_param_diff();
        lsolver.net.empty_net_param_diff();
        gsolver.net.set_input_dim([0, 2, fea_sz(3), fea_sz(2), fea_sz(1)]);
        lsolver.net.set_input_dim([0, 2, fea_sz(3), fea_sz(2), fea_sz(1)]);
        
        fea2_train_l{1}(:,:,:,1) = lfea1;
        fea2_train_l{1}(:,:,:,2) = lfea2;
        fea2_train_g{1}(:,:,:,1) = gfea1;
        fea2_train_g{1}(:,:,:,2) = gfea2;
        
        
        l_pre_map = lsolver.net.forward(fea2_train_l);
        g_pre_map = gsolver.net.forward(fea2_train_g);

        diff_l{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
        diff_l{1}(:,:,:,2) = 0.5*(l_pre_map{1}(:,:,:,2)-permute(single(map2), [2,1,3]));
        %
        diff_g{1}(:,:,:,1) = 0.5*(g_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
        diff_g{1}(:,:,:,2) = 0.5*(g_pre_map{1}(:,:,:,2)-permute(single(map2), [2,1,3]));
        
        lsolver.net.backward(diff_l);
        gsolver.net.backward(diff_g);
        lsolver.apply_update();
        gsolver.apply_update();
        gsolver.net.set_input_dim([0, 1, fea_sz(3), fea_sz(2), fea_sz(1)]);
        lsolver.net.set_input_dim([0, 1, fea_sz(3), fea_sz(2), fea_sz(1)]);
        
        %     lnet_output.set_diff(l_diff);
        %     gnet_output.set_diff(g_diff);
        %     lsolver.net.backward_prefilled();
        %     gsolver.net.backward_prefilled();
%         lsolver.net.backward({l_diff});
%         gsolver.net.backward({g_diff});
%         lsolver.apply_update();
%         gsolver.apply_update();
          
%         caffe('set_phase_train');
%         caffe('reshape_input', 'lsolver', [0, 2, length(lid), fea_sz(2), fea_sz(1)]);
%          caffe('reshape_input', 'gsolver', [0, 2, length(gid), fea_sz(2), fea_sz(1)]);
%         fea2_train_l{1}(:,:,:,1) = lfea1;
%         fea2_train_l{1}(:,:,:,2) = lfea2;
%         
%         fea2_train_g{1}(:,:,:,1) = gfea1;
%         fea2_train_g{1}(:,:,:,2) = gfea2;
%         %             l_pre_map = caffe('forward_lnet', {fea2_store});
%         l_pre_map = caffe('forward_lnet', fea2_train_l);
%         g_pre_map = caffe('forward_gnet', fea2_train_g);
%         %             diff = l_pre_map{1}-permute(single(map2_store), [2,1,3]);
%         diff_l{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
%         diff_l{1}(:,:,:,2) = 0.5*(l_pre_map{1}(:,:,:,2)-permute(single(map2), [2,1,3]));
%         %
%         diff_g{1}(:,:,:,1) = 0.5*(g_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
%         diff_g{1}(:,:,:,2) = 0.5*(g_pre_map{1}(:,:,:,2)-permute(single(map2), [2,1,3]));
%         %         diff = permute((l_pre_map-single(map)).*single(map<=0), [2,1,3]);
%         caffe('backward_lnet', diff_l);
%         caffe('update_lnet');
%         caffe('backward_gnet', diff_g);
%         caffe('update_gnet');
%         caffe('reshape_input', 'lsolver', [0, 1, length(lid), fea_sz(2), fea_sz(1)]);
%         caffe('reshape_input', 'gsolver', [0, 1, length(gid), fea_sz(2), fea_sz(1)]);
    end
    %% update scale estimator
        % extract the training sample feature map for the scale filter
    base_location = [pos(1), pos(2), target_sz(1), target_sz(2)];
%     base_location = [GT(im2_id, 1:2) + floor(GT(im2_id, 3:4)/2), target_sz(1), target_sz(2)];
    xs = get_scale_sample(im2, base_location, scale_param.scaleFactors, scale_param.scale_window, scale_param.scale_model_sz);
    
    % calculate the scale filter update
    xsf = fft(xs,[],2);
    new_sf_num = bsxfun(@times, scale_param.ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    
    
    if mod(im2_id, 1) == 0
        scale_param.sf_den = (1 - scale_param.learning_rate) * scale_param.sf_den + scale_param.learning_rate * new_sf_den;
        scale_param.sf_num = (1 - scale_param.learning_rate) * scale_param.sf_num + scale_param.learning_rate * new_sf_num;
    end
    
    
    
    
    %% Drwa resutls
    if im2_id == im1_id,  %first frame, create GUI
        figure('Number','off', 'Name','Tracking Results');
        im_handle = imshow(uint8(im2), 'Border','tight', 'InitialMag', 100 + 100 * (length(im2) < 500));
        rect_handle = rectangle('Position', location, 'EdgeColor','r', 'linewidth', 2);
        text_handle = text(10, 10, sprintf('#%d / %d',im2_id, fnum));
        set(text_handle, 'color', [0 1 1], 'fontsize', 15);
    else
        set(im_handle, 'CData', uint8(im2))
        set(rect_handle, 'Position', location)
        set(text_handle, 'string', sprintf('#%d / %d',im2_id, fnum));
    end
    
    drawnow
%       location = GT(im2_id, :);
    
    
end
save([track_res '/position.mat'], 'position');
fprintf('Speed: %d fps\n', fnum/t);
end

function [roi, roi_pos, preim, pad] = ext_roi(im, GT, l_off, roi_size, r_w_scale)
[h, w, ~] = size(im);
win_w = GT(3);
win_h = GT(4);
win_lt_x = GT(1);
win_lt_y = GT(2);
win_cx = round(win_lt_x+win_w/2+l_off(1));
win_cy = round(win_lt_y+win_h/2+l_off(2));
roi_w = r_w_scale(1)*win_w;
roi_h = r_w_scale(2)*win_h;
x1 = win_cx-round(roi_w/2);
y1 = win_cy-round(roi_h/2);
x2 = win_cx+round(roi_w/2);
y2 = win_cy+round(roi_h/2);

im = double(im);
clip = min([x1,y1,h-y2, w-x2]);
pad = 0;
if clip<=0
    pad = abs(clip)+1;
    im = padarray(im, [pad, pad]);
    x1 = x1+pad;
    x2 = x2+pad;
    y1 = y1+pad;
    y2 = y2+pad;
end
roi =  imresize(im(y1:y2, x1:x2, :), [roi_size, roi_size]);
preim = zeros(size(im,1), size(im,2));
roi_pos = [x1, y1, x2-x1+1, y2-y1+1];
% marginl = floor((roi_warp_size-roi_size)/2);
% marginr = roi_warp_size-roi_size-marginl;

% roi = roi(marginl+1:end-marginr, marginl+1:end-marginr, :);
% roi = imresize(roi, [roi_size, roi_size]);
end


function I = impreprocess(im)
mean_pix = [103.939, 116.779, 123.68]; % BGR
im = permute(im, [2,1,3]);
im = im(:,:,3:-1:1);
I(:,:,1) = im(:,:,1)-mean_pix(1); % substract mean
I(:,:,2) = im(:,:,2)-mean_pix(2);
I(:,:,3) = im(:,:,3)-mean_pix(3);
end

function map =  GetMap(im_sz, fea_sz, roi_size, location, l_off, s, type)
if strcmp(type, 'box')
    map = ones(im_sz);
    map = crop_bg(map, location, [0,0,0]);
elseif strcmp(type, 'gaussian')
    
    map = zeros(im_sz(1), im_sz(2));
    scale = min(location(3:4))/3;
    %     mask = fspecial('gaussian', location(4:-1:3), scale);
    mask = fspecial('gaussian', min(location(3:4))*ones(1,2), scale);
    mask = imresize(mask, location(4:-1:3));
    mask = mask/max(mask(:));
    
    x1 = location(1);
    y1 = location(2);
    x2 = x1+location(3)-1;
    y2 = y1+location(4)-1;
    
    clip = min([x1,y1,im_sz(1)-y2, im_sz(2)-x2]);
    pad = 0;
    if clip<=0
        pad = abs(clip)+1;
        map = zeros(im_sz(1)+2*pad, im_sz(2)+2*pad);
        %         map = padarray(map, [pad, pad]);
        x1 = x1+pad;
        x2 = x2+pad;
        y1 = y1+pad;
        y2 = y2+pad;
    end
    
    
    map(y1:y2,x1:x2) = mask;
    if clip<=0
        map = map(pad+1:end-pad, pad+1:end-pad);
    end
    
elseif strcmp(type, 'trans_gaussian')
    sz = location([4,3]);
    output_sigma_factor = 1/32;%1/16;
    [rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
    output_sigma = sqrt(prod(location([3,4]))) * output_sigma_factor;
    mask = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    map = zeros(im_sz(1), im_sz(2));
    
    x1 = location(1);
    y1 = location(2);
    x2 = x1+location(3)-1;
    y2 = y1+location(4)-1;
    
    clip = min([x1,y1,im_sz(1)-y2, im_sz(2)-x2]);
    pad = 0;
    if clip<=0
        pad = abs(clip)+1;
        map = zeros(im_sz(1)+2*pad, im_sz(2)+2*pad);
        %         map = padarray(map, [pad, pad]);
        x1 = x1+pad;
        x2 = x2+pad;
        y1 = y1+pad;
        y2 = y2+pad;
    end
    map(y1:y2,x1:x2) = mask;
else error('unknown map type');
end
map = ext_roi(map(1+pad:end-pad, 1+pad:end-pad), location, l_off, roi_size, s);
map = imresize(map(:,:,1), [fea_sz(1), fea_sz(2)]);
end

function I = crop_bg(im, GT, mean_pix)
[im_h, im_w, ~] = size(im);
win_w = GT(3);
win_h = GT(4);
win_lt_x = max(GT(1), 1);
win_lt_x = min(im_w, win_lt_x);
win_lt_y = max(GT(2), 1);
win_lt_y = min(im_h, win_lt_y);

win_rb_x = max(win_lt_x+win_w-1, 1);
win_rb_x = min(im_w, win_rb_x);
win_rb_y = max(win_lt_y+win_h-1, 1);
win_rb_y = min(im_h, win_rb_y);

I = zeros(im_h, im_w, 3);
I(:,:,1) = mean_pix(3);
I(:,:,2) = mean_pix(2);
I(:,:,3) = mean_pix(1);
I(win_lt_y:win_rb_y, win_lt_x:win_rb_x, :) = im(win_lt_y:win_rb_y, win_lt_x:win_rb_x, :);
end

function param = loc2affgeo(location, p_sz)
% location = [tlx, tly, w, h]

cx = location(1)+(location(3)-1)/2;
cy = location(2)+(location(4)-1)/2;
param = [cx, cy, location(3)/p_sz, 0, location(4)/location(3), 0]';

end


function   location = affgeo2loc(geo_param, p_sz)
w = geo_param(3)*p_sz;
h = w*geo_param(5);
tlx = geo_param(1) - (w-1)/2;
tly = geo_param(2) - (h-1)/2;
location = round([tlx, tly, w, h]);
end


function geo_params = drawparticals(geo_param, pf_param)
geo_param = repmat(geo_param, [1,pf_param.p_num]);
geo_params = geo_param + randn(6,pf_param.p_num).*repmat(pf_param.affsig(:),[1,pf_param.p_num]);
end


function drawresult(fno, frame, sz, mat_param)
figure(1); clf;
set(gcf,'DoubleBuffer','on','MenuBar','none');
colormap('gray');
axes('position', [0 0 1 1])
imagesc(frame, [0,1]); hold on;
text(5, 18, num2str(fno), 'Color','y', 'FontWeight','bold', 'FontSize',18);
drawbox(sz(1:2), mat_param, 'Color','r', 'LineWidth',2.5);
axis off; hold off;
drawnow;
end

function [sal, id] = compute_saliency(fea1, map, solver)
caffe('set_phase_test');
if strcmp(solver, 'lsolver')
    out = caffe('forward_lnet', fea1);
    diff1 = {out{1}-permute(single(map), [2,1,3])};
    input_diff1 = caffe('backward_lnet', diff1);
    diff2 = {single(ones(size(fea1{1},1)))};
    input_diff2 = caffe('backward2_lnet', diff2);
elseif strcmp(solver, 'gsolver')
    out = caffe('forward_gnet', fea1);
    diff2 = {single(ones(size(fea1{1},1)))};
    diff1 = {out{1}-permute(single(map), [2,1,3])};
    input_diff1 = caffe('backward_gnet', diff1);
    input_diff2 = caffe('backward2_gnet', diff2);
else
    error('Unkonwn solver type')
end
% sal = sum(sum(input_diff2{1}.*(fea1{1}).^2));
% sal = -sum(sum(input_diff1{1}.*fea1{1}))+0.5*sum(sum(input_diff2{1}.*(fea1{1}).^2));
sal = -sum(sum(input_diff1{1}.*fea1{1}+0.5*input_diff2{1}.*(fea1{1}).^2));

sal = sal(:);
[~, id] = sort(sal, 'descend');
end
