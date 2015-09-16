seq_path = '~/Downloads/PF_CNN_SVM/data/';
seq = dir(seq_path);
train_path = 'data/train/';
val_path = 'data/val/';
if ~isdir(train_path)
    mkdir(train_path);
end
if ~isdir(val_path)
    mkdir(val_path);
end
delete([train_path '*'])

location_off = [];
location_off_cell = {};
%% load data
for seq_id = 1:length(seq)
    if strcmp(seq(seq_id).name, '.') || strcmp(seq(seq_id).name, '..') || strcmp(seq(seq_id).name, '..') ...
            || strcmp(seq(seq_id).name, 'AE_train_Deer') || ~isdir([seq_path seq(seq_id).name])
        continue;
    end
    location = load([seq_path seq(seq_id).name '/groundtruth_rect.txt']);
%     location = location(end:-1:1, :);
    off = location(2:end, 1:2) - location(1:end-1, 1:2);
    location_off = [location_off; off];
    location_off_cell = {location_off_cell{:}, off};

    location = location(end:-1:1, :);
    off = location(2:end, 1:2) - location(1:end-1, 1:2);
    location_off = [location_off; off];
    location_off_cell = {location_off_cell{:}, off};
end

%% compute mean and varaition
off_mean = mean(location_off, 1);
off_std = std(location_off);
%% normalize data by substracting mean and normailizing by std
for seq_id = 1:length(location_off_cell)
    location_off_cell{seq_id} = bsxfun(@minus, location_off_cell{seq_id}, off_mean);
%     location_off_cell{seq_id} = bsxfun(@rdivide,location_off_cell{seq_id}, off_std);
end
%% generate cont, the first element of cont for each sequence is 0 others are all 1;
cont = {};
for seq_id = 1:length(location_off_cell)
    cont{seq_id} = ones(length(location_off_cell{seq_id}), 1);
    cont{seq_id}(1) = 0;
end

%% write training samples into hdf5 files

time_step = 60; batch_size = 10; id_in_batch = 1; 
data = nan(time_step, 2, batch_size);
label = nan(time_step, 2, batch_size);
data_cont = nan(time_step, batch_size);
id = 0;
for seq_id = 1:length(location_off_cell)
    samp_num = 1;
    rand_key = rand(1);
    for step = 1:size(location_off_cell{seq_id}, 1) - time_step
        data(:, :, id_in_batch) = location_off_cell{seq_id}(step:step+time_step - 1, :);
        label(:, :, id_in_batch) = location_off_cell{seq_id}(step+1:step+1+time_step - 1, :);
        data_cont(:, id_in_batch) = cont{seq_id}(step:step+time_step - 1);
        
        id_in_batch = id_in_batch + 1;
        if id_in_batch > batch_size       
            file_name = sprintf('%s%03d-%03d.h5', train_path, seq_id, samp_num);
%             file_name = sprintf('%s%03d-%03d-%03d.h5', train_path, randi(100), randi(100), id);
            if rand_key > 0.4
                file_name = sprintf('%strain-%03d-%03d.h5', train_path, seq_id, samp_num);
            else
                file_name = sprintf('%stest-%03d-%03d.h5', train_path, seq_id, samp_num);
            end

            data_cont(1, :) = 0;
            h5create(file_name,'/data',[2, batch_size, time_step],'ChunkSize',[2, batch_size, 10],'Datatype','single', 'Deflate',9);
            h5create(file_name,'/label',[2, batch_size, time_step],'ChunkSize',[2, batch_size, 10],'Datatype','single', 'Deflate',9);
            h5create(file_name,'/cont',[batch_size, time_step],'ChunkSize',[batch_size 10],'Datatype','single', 'Deflate',9);
            %
            h5write(file_name, '/data', permute(single(data), [2,3,1]));
            h5write(file_name, '/label', permute(single(label), [2,3,1]));
            h5write(file_name, '/cont', permute(single(data_cont), [2,1]));
            id_in_batch = 1;
            samp_num = samp_num + 1;
            id = id + 1;
        end
    end
end

    % write the last batch
    if id_in_batch ~= 1
        file_name = sprintf('%stest-%03d-%010d.h5', train_path, seq_id, samp_num);
        
        if rand_key > 0.4
            file_name = sprintf('%strain-%03d-%03d.h5', train_path, seq_id, samp_num);
        else
            file_name = sprintf('%stest-%03d-%03d.h5', train_path, seq_id, samp_num);
        end
%          file_name = sprintf('%s%03d-%03d.h5', train_path, randi(100), randi(100));
        h5create(file_name,'/data',[2, batch_size, time_step],'ChunkSize',[2, batch_size, 10],'Datatype','single', 'Deflate',9);
        h5create(file_name,'/label',[2, batch_size, time_step],'ChunkSize',[2, batch_size, 10],'Datatype','single', 'Deflate',9);
        h5create(file_name,'/cont',[batch_size, time_step],'ChunkSize',[10 10],'Datatype','single', 'Deflate',9);
        %
        h5write(file_name, '/data', permute(single(data), [2,3,1]));
        h5write(file_name, '/label', permute(single(label), [2,3,1]));
        h5write(file_name, '/cont', permute(single(data_cont), [2,1]));
    end







