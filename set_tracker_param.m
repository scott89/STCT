pf_param = struct('affsig', [10,10,.004,.00,0.00,0], 'p_sz', 64,...
            'p_num', 600, 'mv_thr', 0.1, 'up_thr', 0.35, 'pos_thr', 0.3, 'roi_scale', 2, 'map_sigma_factor', 1/32); % roi_scale = 2;
   
close all;

track_res = ['benchmark_res/'];
sample_res = ['sample_res/' set_name '/'];
if ~isdir(track_res)
    mkdir(track_res);
end

if ~isdir(sample_res)
    mkdir(sample_res);
end

fprintf('epsilon: %f\n', epsilon);

data_path = ['~/Downloads/PF_CNN_SVM/data/' set_name '/'];
GT = load([data_path 'groundtruth_rect.txt']);
dia = (GT(im1_id, 3)^2+GT(im1_id, 4)^2)^0.5;

% scale = gt(1, 3)/ gt (1, 4);
scale = [dia/GT(im1_id, 3), dia/GT(im1_id, 4)];
l1_off = [0,0];
l2_off = [0,0];
s1 = pf_param.roi_scale*[scale(1),scale(2)];
s2 = pf_param.roi_scale*[scale(1),scale(2)];
%% init caffe
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
%% fsolver init
% feature_solver_def_file = 'feature_solver.prototxt';
% model_file = 'VGG_ILSVRC_16_layers.caffemodel';
% fsolver = caffe.Solver(feature_solver_def_file);
% fsolver.net.copy_from(model_file);
% %% Qnet solver init
% Qnet_solver_def_file = 'solver/Qnet_solver.prototxt';
% 
% Qnet_model_path = 'Qnet_model';
% Qnet_model_file = ['Qnet_model/Qnet_model.caffemodel'];
% Qsolver = caffe.Solver(Qnet_solver_def_file);
% if exist(Qnet_model_file, 'file')
%     Qsolver.net.copy_from(Qnet_model_file);
% end
% %% Qsolver for generating target
% Qtsolver = caffe.Solver(Qnet_solver_def_file);
fsolver = Qtfsolver.fsolver;
Qsolver = Qtfsolver.Qsolver;
Qtsolver = Qtfsolver.Qtsolver;
%% gnet solver init
gnet_solver_def_file = ['solver/gnet_solver_' num2str(ch_num) '.prototxt'];
gsolver = caffe.Solver(gnet_solver_def_file);
%% lnet solver init
lnet_solver_def_file = ['solver/lnet_solver_' num2str(ch_num) '.prototxt']; 
lsolver = caffe.Solver(lnet_solver_def_file);
%%
Qnet_model_path = 'Qnet_model/';
train_data_path = 'data/';
%%
snapshot = 1000;
Qtupdate_interval = 10000;
forget_rate = 0.9;
batch_size = 32;
buffer_size = 10000;
rl_channel_num = 3;
%% 
roi_size = 361;%368; %380;

mean_pix = [103.939, 116.779, 123.68]; 

% fnum = 20;
location = GT(im1_id,:);
% location = GT(1,:);
pf_param.ratio = location(3)/pf_param.p_sz;
pf_param.affsig(3) = pf_param.affsig(3)*pf_param.ratio;
pf_param.affsig_o = pf_param.affsig;
pf_param.affsig(3) = 0;
pf_param.minconf = 0.5;
