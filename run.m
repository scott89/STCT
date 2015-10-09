data_path = '~/Downloads/PF_CNN_SVM/data/';
dataset = dir(data_path);
% 
% epsilon = 0.5105;
% epsilon = 0.01;
% epsilon = 0.3306;
epsilon = 1;

% iter_num = 487375;
% iter_num = 666560;
% generate_data_list;
% iter_num = 1150000;
iter_num = 0;
buffer_sz = 10000;
% buffer_sz = 1;
generate_data_list
caffe.reset_all;
Qtfsolver = init_Qtfsolver(iter_num);
data_empty = true;
for k = 1:300
    for i= 1:length(dataset)
        if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
            continue;
        end
        track_success = false;
        [track_success, iter_num, epsilon, data_list, data_empty, restart_frame, lfea1, map1]= cnn2_pf_tracker(dataset(i).name, 1, 512, epsilon, iter_num, Qtfsolver, data_list, data_empty);
        fail_num = 0;
        while ~track_success
            [track_success, iter_num, epsilon, data_list, data_empty, restart_frame, lfea1, map1]= re_tracker(dataset(i).name, restart_frame, 512, epsilon, iter_num, Qtfsolver, data_list, data_empty, lfea1, map1);
            fail_num = fail_num + 1;
            if mod(fail_num, 10) && restart_frame > 1
                restart_frame = restart_frame - 10;
            elseif fail_num == 100
                    break;  
            end
        end

        if epsilon > 0.1
            epsilon = epsilon - 1e-6;
        end
    end
end

caffe.reset_all;
% for i=31:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 512);
% % end
% for i=51:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 256);
% end
% 
% 
% 
% 
% 
% for i=1:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 128);
% end
% 
% for i=1:length(dataset)
%     if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
%         continue;
%     end
%     cnn2_pf_tracker(dataset(i).name, 1, 64);
% end% 
% % iter = 50;
% for i=1:iter
%     l_pre_map = caffe('forward_lnet', {lfea2_train});
%     diff{1}(:,:,:,1) = 0.5*(l_pre_map{1}(:,:,:,1)-permute(single(map2_store), [2,1,3]));
%     diff{1}(:,:,:,2) = 0.3*squeeze(l_pre_map{1}(:,:,:,2)-permute(single(map),[2,1,3])).*permute(single(map<=0), [2,1,3]);
%     caffe('backward_lnet', diff);
%     caffe('update_lnet');
%     %                     l_pre_map = caffe('forward_lnet', {lfea2});
%     %                     %         diff = permute(l_pre_map-single(map), [2,1,3]);
%     %                     %                     diff = l_pre_map{1}-single(l_pre_map_o{1}>0.01).*single(permute(map>0, [2,1,3]));
%     %                     diff = (l_pre_map-single(map)).*permute(single(map<=0), [2,1,3]);
%     %                     caffe('backward_lnet', {diff});
%     %                     caffe('update_lnet');
%     figure(50); subplot(1,2,1); imagesc(permute(l_pre_map{1}(:,:,:,1), [2,1,3]));
%     figure(50); subplot(1,2,2); imagesc(permute(l_pre_map{1}(:,:,:,2), [2,1,3]));
% end