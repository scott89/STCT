
para_3_diff = fsolver.net.params('conva_3',1).get_diff();
para_2_diff = fsolver.net.params('conva_2',1).get_diff();
para_1_3_diff = fsolver.net.params('conva_1_3',1).get_diff();
para_1_4_diff = fsolver.net.params('conva_1_4',1).get_diff();
para_1_5_diff = fsolver.net.params('conva_1_5',1).get_diff();
para_1_5_down_diff = fsolver.net.params('conva_1_5_down',1).get_diff();


para_3 = fsolver.net.params('conva_3',1).get_data();
para_2 = fsolver.net.params('conva_2',1).get_data();
para_1_3 = fsolver.net.params('conva_1_3',1).get_data();
para_1_4 = fsolver.net.params('conva_1_4',1).get_data();
para_1_5 = fsolver.net.params('conva_1_5',1).get_data();
para_1_5_down = fsolver.net.params('conva_1_5_down',1).get_data();

blob_3_diff = fsolver.net.blobs('conva_3').get_diff();
blob_2_diff = fsolver.net.blobs('conva_2').get_diff();
blob_1_3_diff = fsolver.net.blobs('conva_1_3').get_diff();
blob_1_4_diff = fsolver.net.blobs('conva_1_4').get_diff();
blob_1_5_diff = fsolver.net.blobs('conva_1_5').get_diff();
blob_1_5_down_diff = fsolver.net.blobs('conva_1_5_down').get_diff();
conv_5_diff = fsolver.net.blobs('conv5_3').get_diff();
conv_4_diff = fsolver.net.blobs('conv4_3').get_diff();
conv_3_diff = fsolver.net.blobs('pool3').get_diff();

blob_3 = fsolver.net.blobs('conva_3').get_data();
blob_2 = fsolver.net.blobs('conva_2').get_data();
blob_1 = fsolver.net.blobs('conva_1').get_data();
blob_1_3 = fsolver.net.blobs('conva_1_3').get_data();
blob_1_4 = fsolver.net.blobs('conva_1_4').get_data();
blob_1_5 = fsolver.net.blobs('conva_1_5').get_data();
blob_1_5_down = fsolver.net.blobs('conva_1_5_down').get_data();
blob_1_norm = fsolver.net.blobs('conva_1_norm').get_data();
conv_5 = fsolver.net.blobs('conv5_3').get_data();
conv_4 = fsolver.net.blobs('conv4_3').get_data();
conv_3 = fsolver.net.blobs('pool3').get_data();






% 
% for ii = 1:10
%     fsolver.net.empty_net_param_diff();
%     pre_heat_map_train = fsolver.net.forward_from_to(middle_layer + 1, last_layer);
%     pre_heat_map_train = pre_heat_map_train{1};
%     diff_cnna2 = 0.5 * (pre_heat_map_train-map2);
%     %
%     fsolver.net.backward_from_to({single(diff_cnna2)}, last_layer, middle_layer + 1);
%     %% first frame
%     %             pre_heat_map_train = fsolver.net.forward({single(roi1(:,:,:,4))});
%     %             pre_heat_map_train = pre_heat_map_train{1};
%     %             diff_cnna1 = 0.5*(pre_heat_map_train-map1(:,:,:,4));
%     %             fsolver.net.backward({diff_cnna1});
%     fsolver.apply_update();
%     figure(55);imshow(permute(pre_heat_map_train(:,:,:,1), [2,1,3]));
%     fprintf('loss: %f\n', sum(abs(diff_cnna2(:))));
% end