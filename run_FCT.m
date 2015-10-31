function [ res] = run_FCT(seq, seq_name, res_path, bSaveImage )

close all
res = cnn2_pf_tracker(seq_name, [seq.startFrame, seq.endFrame], seq.init_rect);


end

