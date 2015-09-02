function  cnn2_pf_tracker(set_name, im1_id, ch_num)
% set_name = 'Shaking'; im1_id = 1; ch_num = 512;
cleanupObj = onCleanup(@cleanupFun);
% rng('default');
% rng(0);
rand('state', 0);
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


%% preprocess roi
roi1 = impreprocess(roi1);
fsolver.net.set_net_phase('test');
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');
feature_blob5 = fsolver.net.blobs('conv5_3');
fsolver.net.set_input_dim([0, 1, 3, roi_size, roi_size]);
feature_input.set_data(single(roi1));
fsolver.net.forward_prefilled();
% fea = fsolver.net.forward({roi1});

% fea1 = caffe('forward', {single(roi1)});
lfea1 = feature_blob4.get_data();
fea_sz = size(lfea1);

gfea1 = imresize(feature_blob5.get_data(), fea_sz(1:2));

cos_win = single(hann(fea_sz(1)) * hann(fea_sz(2))');
lfea1 = bsxfun(@times, lfea1, cos_win);
gfea1 = bsxfun(@times, gfea1, cos_win);
%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

max_iter = 50;
map1 =  GetMap(size(im1), fea_sz, roi_size, location, l1_off, s1, pf_param.map_sigma_factor, 'trans_gaussian');
%% Init Scale Estimator
scale_param = init_scale_estimator;
scale_param.train_sample = get_scale_sample(lfea1, scale_param.scaleFactors_train, scale_param.scale_window_train);

%% train


lsolver.net.set_net_phase('train');
gsolver.net.set_net_phase('train');
lnet_input = lsolver.net.blobs('data');
gnet_input = gsolver.net.blobs('data');
lnet_output = lsolver.net.blobs('conv5_f2');
gnet_output = gsolver.net.blobs('conv5_f2');

%% Iterations
figure(11);stem(scale_param.y);
diff_mask = ones(1, scale_param.number_of_scales_train);
diff_mask((scale_param.number_of_scales_train+1)/2) = 0;
gsolver.net.set_input_dim([0, scale_param.number_of_scales_train, fea_sz(3), fea_sz(2), fea_sz(1)]);
for i=1:max_iter
    gsolver.net.empty_net_param_diff();
    lsolver.net.empty_net_param_diff();

    l_pre_map = lsolver.net.forward({lfea1});
    scale_score = gsolver.net.forward({scale_param.train_sample});
    l_pre_map = l_pre_map{1};
    scale_score = scale_score{1};
    figure(1011); subplot(1,2,1); imagesc(permute(l_pre_map,[2,1,3]));
    figure(1011); subplot(1,2,2); stem(scale_score)
     
    l_diff = l_pre_map-permute(single(map1), [2,1,3]);
%     scale_score = scale_score - max(scale_score);
%     p = exp(scale_score) / sum(exp(scale_score));
%     figure(1011); subplot(1,2,2); stem(p)
    
%     g_diff = p;
%     g_diff(17) = 1 - g_diff(17);
    g_diff = (1*(scale_score-scale_param.y) - 0*min(scale_score((scale_param.number_of_scales_train+1)/2) - scale_score - 0.5, 0) .* diff_mask)/length(scale_param.number_of_scales_train);

%     g_diff = (0.0001*(scale_score-scale_param.y) + scale_score(17)-1 - 2*min(scale_score(17) - scale_score - 0.5, 0) .* diff_mask)/length(scale_param.number_of_scales);
   
    lsolver.net.backward({single(l_diff)});
    gsolver.net.backward({single(g_diff)});
    lsolver.apply_update();
    gsolver.apply_update();
    fprintf('Iteration %03d/%03d, Local Loss %f, Global Loss %f\n', i, max_iter, sum(abs(l_diff(:))), sum(abs(g_diff(:))));
end
%% ================================================================


t=0;
fnum = size(GT,1);
positions = zeros(fnum, 4);
close all
gsolver.net.set_input_dim([0, scale_param.number_of_scales_test, fea_sz(3), fea_sz(2), fea_sz(1)]);

for im2_id = im1_id:fnum
    tic;
    lsolver.net.set_net_phase('test');
    gsolver.net.set_net_phase('test');
    location_last = location;
    tic
    fprintf('Processing Img: %d/%d ', im2_id, fnum);
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
    l_pre_map = permute(l_pre_map{1}, [2,1,3])/(max(l_pre_map{1}(:))+eps);
    
%     
%     
%     g_pre_map = gsolver.net.forward({gfea2});
%     g_pre_map_train = g_pre_map{1};
%     g_pre_map = permute(g_pre_map{1}, [2,1,3])/(max(g_pre_map{1}(:))+eps);
    
    
 
    %% compute local confidence
    l_roi_map = imresize(l_pre_map, roi_pos(4:-1:3));
    l_im_map = padded_zero_map;
    l_im_map(roi_pos(2):roi_pos(2)+roi_pos(4)-1, roi_pos(1):roi_pos(1)+roi_pos(3)-1) = l_roi_map;
    l_im_map = l_im_map(pad+1:end-pad, pad+1:end-pad);
    [l_y, l_x] = find(l_im_map == max(l_im_map(:)), 1);

    %% local scale estimation
    
    base_location_l = [l_x - location(3)/2, l_y - location(4)/2, location([3,4])];

    roi2 = ext_roi(im2, base_location_l, l2_off,  roi_size, s2);
    roi2 = impreprocess(roi2);
    feature_input.set_data(single(roi2));
    fsolver.net.forward_prefilled();  
    lfea2 = feature_blob4.get_data();
    lfea2 = bsxfun(@times, lfea2, cos_win);
    scale_sample = get_scale_sample(lfea2, scale_param.scaleFactors_test, scale_param.scale_window_test);
    scale_score = gsolver.net.forward({scale_sample});
    scale_score = scale_score{1};
    [~, recovered_scale]= max(scale_score);
%     if im2_id <= 10 && max_scale_score < 0.5 || max_scale_score < 0.3
%         recovered_scale = (scale_param.number_of_scales_test+1)/2;
%     end
    recovered_scale = scale_param.number_of_scales_test+1 - recovered_scale;
    % update the scale
    scale_param.currentScaleFactor = scale_param.scaleFactors_test(recovered_scale);
    target_sz = location([3, 4]) * scale_param.currentScaleFactor;
    target_sz = round(target_sz);
    location = [l_x - round(target_sz(1)/2), l_y - round(target_sz(2)/2), target_sz(1), target_sz(2)];
    t = t + toc;
    fprintf(' scale = %f\n', scale_param.scaleFactors_test(recovered_scale));
    %% show results
      if im2_id == im1_id
       figure('Number','off', 'Name','Target Heat Maps');
       subplot(1,2,1);
       im_handle1 = imagesc(l_pre_map);
       subplot(1,2,2);
       stem(scale_score);
       im_handle2 = gca;
    else
       set(im_handle1, 'CData', l_pre_map)
       stem(im_handle2, scale_score);
      end   
    
    %% Update lnet and gnet
    if    recovered_scale ~= (scale_param.number_of_scales_test+1)/2 %|| im2_id <= 3%mod(im2_id, 1) == 0
%         l_off = location_last(1:2)-location(1:2);
%         map2 = GetMap(size(im2), fea_sz, roi_size, floor(location), floor(l_off), s2, pf_param.map_sigma_factor, 'trans_gaussian');
        
        roi2 = ext_roi(im2, location, l2_off,  roi_size, s2);
        roi2 = impreprocess(roi2);
        feature_input.set_data(single(roi2));
        fsolver.net.forward_prefilled();
        lfea2 = feature_blob4.get_data();
        lfea2 = bsxfun(@times, lfea2, cos_win);
        map2 = GetMap(size(im2), fea_sz, roi_size, floor(location), floor(l1_off), s2, pf_param.map_sigma_factor, 'trans_gaussian');
     
        gsolver.net.set_input_dim([0, scale_param.number_of_scales_train, fea_sz(3), fea_sz(2), fea_sz(1)]);
        lsolver.net.set_net_phase('train');
        gsolver.net.set_net_phase('train');
        gsolver.net.empty_net_param_diff();
        lsolver.net.empty_net_param_diff();
%         gsolver.net.set_input_dim([0, 2, fea_sz(3), fea_sz(2), fea_sz(1)]);
        lsolver.net.set_input_dim([0, 2, fea_sz(3), fea_sz(2), fea_sz(1)]);
        
        fea2_train_l{1}(:,:,:,1) = lfea1;
        fea2_train_l{1}(:,:,:,2) = lfea2;
        
        l_pre_map = lsolver.net.forward(fea2_train_l);
        
        
        scale_param.train_sample = get_scale_sample(lfea2, scale_param.scaleFactors_train, scale_param.scale_window_train);
        scale_score = gsolver.net.forward({scale_param.train_sample});
        scale_score = scale_score{1};
        diff_g = (1*(scale_score-scale_param.y) - 0*min(scale_score((scale_param.number_of_scales_train+1)/2) - scale_score - 0.5, 0) .* diff_mask)/length(scale_param.number_of_scales_train);
        diff_g = {single(diff_g)};



        diff_l{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
        diff_l{1}(:,:,:,2) = 0.5*(l_pre_map{1}(:,:,:,2)-permute(single(map2), [2,1,3]));

        lsolver.net.backward(diff_l);
        gsolver.net.backward(diff_g);
        lsolver.apply_update();
        gsolver.apply_update();
%         gsolver.net.set_input_dim([0, 1, fea_sz(3), fea_sz(2), fea_sz(1)]);
        lsolver.net.set_input_dim([0, 1, fea_sz(3), fea_sz(2), fea_sz(1)]);
        gsolver.net.set_input_dim([0, scale_param.number_of_scales_test, fea_sz(3), fea_sz(2), fea_sz(1)]);
    elseif rand(1)>0.5
%         roi2 = ext_roi(im2, location, l2_off,  roi_size, s2);
%         roi2 = impreprocess(roi2);
%         feature_input.set_data(single(roi2));
%         fsolver.net.forward_prefilled();
%         lfea2 = feature_blob4.get_data();
%         lfea2 = bsxfun(@times, lfea2, cos_win);
        map2 = GetMap(size(im2), fea_sz, roi_size, floor(location), floor(l1_off), s2, pf_param.map_sigma_factor, 'trans_gaussian');
        
        lsolver.net.set_net_phase('train');
        lsolver.net.empty_net_param_diff();
%         gsolver.net.set_input_dim([0, 2, fea_sz(3), fea_sz(2), fea_sz(1)]);
        lsolver.net.set_input_dim([0, 2, fea_sz(3), fea_sz(2), fea_sz(1)]);
        
        fea2_train_l{1}(:,:,:,1) = lfea1;
        fea2_train_l{1}(:,:,:,2) = lfea2;
        
        l_pre_map = lsolver.net.forward(fea2_train_l);
        diff_l{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map1), [2,1,3]));
        diff_l{1}(:,:,:,2) = 0.5*(l_pre_map{1}(:,:,:,2)-permute(single(map2), [2,1,3]));

        lsolver.net.backward(diff_l);
        lsolver.apply_update();
        lsolver.net.set_input_dim([0, 1, fea_sz(3), fea_sz(2), fea_sz(1)]);
    end

    
    positions(im2_id, :) = location;
    
    
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
    saveas(im_handle, sprintf([sample_res '%04d.png'], im2_id));
    
    drawnow
%       location = GT(im2_id, :);
    
    
end
results{1}.type = 'rect';
results{1}.res = positions;
results{1}.startFrame = 1;
results{1}.annoBegin = 1;
resutls{1}.len = fnum;


save([track_res  lower(set_name) '_fct_scale_base1.mat'], 'results');
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

function map =  GetMap(im_sz, fea_sz, roi_size, location, l_off, s, output_sigma_factor, type)
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
%     output_sigma_factor = 1/32;%1/16;
   % [rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));

    [rs, cs] = ndgrid((0:sz(1)-1) - floor(sz(1)/2), (0:sz(2)-1) - floor(sz(2)/2));
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
