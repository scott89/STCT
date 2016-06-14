close all;
cur_path = '/home/lijun/Research/Code/tracker_benchmark_v1.0/trackers/STCT/';
addpath([cur_path 'caffe/matlab/'], [cur_path 'util']);

% data_path = [seq.path seq.name '/'];
%% init caffe %%
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
feature_solver_def_file = [cur_path 'model/feature_solver.prototxt'];
model_file = '/home/lijun/Research/Code/FCT_scale_base/model/VGG_ILSVRC_16_layers.caffemodel';


fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);
%% spn solver
spn_solver_def_file =[cur_path 'model/spn_solver.prototxt'];
spn = caffe.Solver(spn_solver_def_file);

max_iter = 80;%150;
mean_pix = [103.939, 116.779, 123.68]; 

%% Init location parameters
dia = (init_rect(3)^2+init_rect(4)^2)^0.5;
rec_scale_factor = [dia/init_rect(3), dia/init_rect(4)];
center_off = [0,0];
roi_scale = 2.5;
roi_scale_factor = roi_scale*[rec_scale_factor(1),rec_scale_factor(2)];
map_sigma_factor = 1/12;
roi_size = 361;

location = init_rect;
%% init ensemble parameters
%% Init scale parameters
scale_param = init_scale_estimator;
%%
