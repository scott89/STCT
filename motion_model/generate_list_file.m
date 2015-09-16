data_path = 'data/train/';
sample = dir([data_path '*h5']);
train_ratio = 0.7;
line = {};
for i = 1 : length(sample)
    line{i} = sprintf('%s%s', data_path, sample(i).name); 
end

train_num = floor(length(sample) * train_ratio);
test_num = length(sample) - train_num;


id = randperm(length(line));
f_id = fopen('hdf5_train_list.txt', 'w+');
for i = 1 : train_num
   fprintf(f_id, '%s\n', line{id(i)}); 
end
fclose(f_id);

f_id = fopen('hdf5_test_list.txt', 'w+');
for i = 1 : test_num
   fprintf(f_id, '%s\n', line{id(i + train_num)}); 
end
fclose(f_id);