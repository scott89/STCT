function results = test_STCT(seq)
cleanupObj = onCleanup(@cleanupFun);
rand('state', 0);
set_tracker_param;

%% read images
num_z = 4;
im1_name = sprintf([data_path 'img/%0' num2str(num_z) 'd.jpg'], im1_id);
im1 = double(imread(im1_name));


if size(im1,3)~=3
    im1(:,:,2) = im1(:,:,1);
    im1(:,:,3) = im1(:,:,1);
end
%% extract roi and display
roi1 = ext_roi(im1, location, center_off,  roi_size, roi_scale_factor);
%% extract vgg feature
roi1 = impreprocess(roi1);
fsolver.net.set_net_phase('test');
feature_input = fsolver.net.blobs('data');
feature_blob4 = fsolver.net.blobs('conv4_3');
fsolver.net.set_input_dim([0, 1, 3, roi_size, roi_size]);
feature_input.set_data(single(roi1));
fsolver.net.forward_prefilled();
deep_feature1 = feature_blob4.get_data();
fea_sz = size(deep_feature1);
cos_win = single(hann(fea_sz(1)) * hann(fea_sz(2))');
deep_feature1 = bsxfun(@times, deep_feature1, cos_win);

scale_param.train_sample = get_scale_sample(deep_feature1, scale_param.scaleFactors_train, scale_param.scale_window_train);
%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% initialization

% if strcmp('Subway', set_name) || strcmp('Crossing', set_name) || strcmp('Skiing', set_name)
%     max_iter = 180;
% end

cnna.net.set_net_phase('train');
spn.net.set_net_phase('train');

spn.net.set_input_dim([0, scale_param.number_of_scales_train, fea_sz(3), fea_sz(2), fea_sz(1)]);
cnna.net.set_input_dim([0, 1, fea_sz(3), fea_sz(2), fea_sz(1)]);

%% prepare training samples
map1 =  GetMap(size(im1), fea_sz, roi_size, location, center_off, roi_scale_factor, map_sigma_factor, 'trans_gaussian');
map1 = permute(map1, [2,1,3]);
map1 = repmat(single(map1), [1,1,ensemble_num]);


%% Iterations
last_loss = 0;
for i=1:max_iter
    spn.net.empty_net_param_diff();
    cnna.net.empty_net_param_diff();
    pre_heat_map1 = cnna.net.forward({deep_feature1, w0});
    scale_score = spn.net.forward({scale_param.train_sample});
    pre_heat_map = pre_heat_map1{1};
    scale_score = scale_score{1};
    diff_cnna = pre_heat_map-map1;
    diff_spn = (scale_score-scale_param.y)/length(scale_param.number_of_scales_train);
    cnna.net.backward({single(diff_cnna)});
    spn.net.backward({single(diff_spn)});
    cnna.apply_update();
    spn.apply_update();
    fprintf('Iteration %03d/%03d, CNN-A Loss %0.1f, SPN Loss %0.1f\n', i, max_iter, sum(abs(diff_cnna(:))), sum(abs(diff_spn(:))));  
    if i == 80 && sum(abs(diff_cnna(:))) - last_loss <= 0
        fid = fopen('list-80.txt', 'a+');
        fprintf(fid, '%s\n', seq.name);
        fclose(fid);
        break;
    elseif i == 80 && sum(abs(diff_cnna(:))) - last_loss > 0
        fid = fopen('list-180.txt', 'a+');
        fprintf(fid, '%s\n', seq.name);
        fclose(fid);
    end
    last_loss = sum(abs(diff_cnna(:)));      
end










