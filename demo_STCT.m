%% demo_STCT
seq_name = 'Basketball';
%% download video data if necessary
if ~exist(['./video/' seq_name '/img/0001.jpg'], 'file')
    system('sh ./video/download_basketball.sh');
end
%% 
seq_info = load(['video/' seq_name '/info.txt']);
seq.name = seq_name;
seq.path = './video/';
seq.startFrame = seq_info(1);
seq.endFrame = seq_info(2);
seq.init_rect = seq_info(3:6);
res = STCT(seq);




