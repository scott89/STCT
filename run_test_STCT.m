function res = run_test_STCT(seq, seq_name, ~, ~)
res=[];
seq.path = '/home/lijun/Research/CVPR16/Data/data/';
seq.name = seq_name;
test_STCT(seq);
return