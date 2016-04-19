%% demo_STCT
seq_name = 'Basketball';
seq_info = load(['video/' seq_name '/info.txt']);
seq.name = seq_name;
seq.path = './video/';
seq.startFrame = seq_info(1);
seq.endFrame = seq_info(2);
seq.init_rect = seq_info(3:6);
res = STCT(seq);




