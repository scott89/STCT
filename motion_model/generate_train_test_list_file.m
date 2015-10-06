data_path = 'data_60/train/';
train_sample = dir([data_path 'train*']);
test_sample = dir([data_path 'test*']);

train_line = {};
test_line = {};
for i = 1 : length(train_sample)
    train_line{i} = sprintf('%s%s', data_path, train_sample(i).name); 
end
for i = 1 : length(test_sample)
    test_line{i} = sprintf('%s%s', data_path, test_sample(i).name); 
end



train_id = randperm(length(train_line));
f_id = fopen('hdf5_train_list_60.txt', 'w+');
for i = 1 : length(train_line)
   fprintf(f_id, '%s\n', train_line{train_id(i)}); 
end
fclose(f_id);

test_id = randperm(length(test_line));
f_id = fopen('hdf5_test_list_60.txt', 'w+');
for i = 1 : length(test_line)
   fprintf(f_id, '%s\n', test_line{test_id(i)}); 
end
fclose(f_id);