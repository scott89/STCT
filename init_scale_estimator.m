function[scale_param] = init_scale_estimator

% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales   
scale_param.scale_sigma_factor = 1/4;%1/4;        % standard deviation for the desired scale filter output
% scale_param.lambda = 1e-2;					% regularization weight (denoted "lambda" in the paper)
% scale_param.learning_rate = 0.025;%0.025;			% tracking model learning rate (denoted "eta" in the paper)
scale_param.number_of_scales_test = 9; %33;%33;%55;%33;           % number of scale levels (denoted "S" in the paper)
scale_param.number_of_scales_train = 33;
scale_param.scale_step = 1.02;%1.02;%1.02;%1.1;%1.02;               % Scale increment factor (denoted "a" in the paper)
    


scale_param.scale_sigma = sqrt(scale_param.number_of_scales_train) * scale_param.scale_sigma_factor;
ss = (1:scale_param.number_of_scales_train) - ceil(scale_param.number_of_scales_train/2);
ys = exp(-0.5 * (ss.^2) / scale_param.scale_sigma^2);
scale_param.y = single(ys);


scale_param.scale_window_train = single(hann(scale_param.number_of_scales_train));
scale_param.scale_window_test = scale_param.scale_window_train((scale_param.number_of_scales_train - scale_param.number_of_scales_test)/2 + 1: (scale_param.number_of_scales_train + scale_param.number_of_scales_test)/2);
%% ++++++++++++++++++++++++++++++++++++++++++++++++++
%     scale_param.scale_window = single(hann(33));
%      scale_param.scale_window = scale_param.scale_window(13:21);
%% ++++++++++++++++++++++++++++++++++++++++++++++++++


% scale factors
ss = 1:scale_param.number_of_scales_train;
scale_param.scaleFactors_train = scale_param.scale_step.^(ceil(scale_param.number_of_scales_train/2) - ss);

%
ss = 1:scale_param.number_of_scales_test;
scale_param.scaleFactors_test = scale_param.scale_step.^(ceil(scale_param.number_of_scales_test/2) - ss);


end

