data_path = '~/Downloads/PF_CNN_SVM/data/';
dataset = dir(data_path);
for i= 5 : length(dataset)
  if ~isdir([data_path dataset(i).name]) || strcmp(dataset(i).name, '.') || strcmp(dataset(i).name, '..') || strcmp(dataset(i).name, 'AE_train_Deer')
    continue;
  end
  cnn2_pf_tracker(dataset(i).name, 1, 512);
end
caffe.reset_all;

