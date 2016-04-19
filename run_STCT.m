function res = run_STCT(seq, seq_name, ~, ~)
res=[];
seq.path = '/home/lijun/Research/CVPR16/Data/data/';
seq.name = seq_name;
res = STCT(seq);
return