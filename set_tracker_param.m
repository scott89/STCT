close all;
addpath('caffe/matlab/', 'util');

% data_path = [seq.path seq.name '/'];
data_path = seq.path(1:end-4);
imgs = dir([data_path 'img/*.jpg']);
im1_id = seq.startFrame;
end_id = seq.endFrame;
sample_res = ['sample_res/' seq.name '/'];
if ~isdir(sample_res)
    mkdir(sample_res);
end
%% init caffe %%
gpu_id = 0;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
feature_solver_def_file = 'model/feature_solver.prototxt';
model_file = '/home/lijun/Research/Code/FCT_scale_base/model/VGG_ILSVRC_16_layers.caffemodel';
% model_file = '/home/lijun/Downloads/vgg16CAM_train_iter_90000.caffemodel';

fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);
%% spn solver
spn_solver_def_file = 'model/spn_solver.prototxt';
spn = caffe.Solver(spn_solver_def_file);
%% cnn-a solver
% cnna_solver_def_file = 'model/cnn-a_solver.prototxt'; 
% cnna = caffe.Solver(cnna_solver_def_file);
max_iter = 80;%150;
mean_pix = [103.939, 116.779, 123.68]; 

%% Init location parameters
dia = (seq.init_rect(3)^2+seq.init_rect(4)^2)^0.5;
rec_scale_factor = [dia/seq.init_rect(3), dia/seq.init_rect(4)];
center_off = [0,0];
roi_scale = 2.5;
roi_scale_factor = roi_scale*[rec_scale_factor(1),rec_scale_factor(2)];
map_sigma_factor = 1/12;
roi_size = 361;

location = seq.init_rect;
%% init ensemble parameters
%% Init scale parameters
scale_param = init_scale_estimator;
%%
