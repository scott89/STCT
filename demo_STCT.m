%% demo_STCT
% seq_name = 'MotorRolling';
seq_name = 'Basketball';
% seq_name = 'Football';
% seq_name = 'Crossing';
% seq_name = 'Subway';
% seq_name = 'Skiing';
init_rect = load(['video/' seq_name '/init_rect.txt']);
res = run_STCT(seq_name, init_rect);




