data_path = 'data/train/';
% data = dir([data_path '*.h5']);
caffe.reset_all;
data_ind = 30008;
x = h5read([data_path data(data_ind).name], '/data');
x = permute(x, [4,1,2,3]);
c = h5read([data_path data(data_ind).name], '/cont');
c = permute(c, [4,3,1,2]);
l = h5read([data_path data(data_ind).name], '/label');
lstm = caffe.Net('lstm_gmm_deploy.prototxt', 'lstm_gmm_model/_iter_170000.caffemodel', 'test');
y = lstm.forward({single(x), single(c)});
alpha = y{1};
[~, ind] = max(alpha);
unique(ind)
caffe.reset_all;


