function Qtfsolver = init_Qtfsolver(iter_num)

caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);


feature_solver_def_file = 'feature_solver.prototxt';
model_file = 'VGG_ILSVRC_16_layers.caffemodel';
fsolver = caffe.Solver(feature_solver_def_file);
fsolver.net.copy_from(model_file);
%% Qnet solver init
Qnet_solver_def_file = 'solver/Qnet_solver.prototxt';

% Qnet_model_file = ['Qnet_model/Qnet_model.caffemodel'];
Qsolver = caffe.Solver(Qnet_solver_def_file);
% if exist(Qnet_model_file, 'file')
%     Qsolver.net.copy_from(Qnet_model_file);
% end
%% Qsolver for generating target
Qtsolver = caffe.Solver(Qnet_solver_def_file);
try
Qsolver.net.copy_from(['Qnet_model/' num2str(iter_num) '.caffemodel']);
Qtsolver.net.copy_from(['Qnet_model/' num2str(0) '.caffemodel']);
catch
end

Qtfsolver.fsolver = fsolver;
Qtfsolver.Qsolver = Qsolver;
Qtfsolver.Qtsolver = Qtsolver;

end