for train_iter = 1:1
    fprintf('========================================================\n');
    for samp_id = im2_id-1 : -1 : im2_id-44
        Qsolver.net.empty_net_param_diff();
        r = -single(reward(samp_id)<-1);
        if samp_id == im2_id-1
            y = -1;
        else
            Q_tp1_all = Qsolver.net.forward({state(:, :, :, samp_id+1)});
            Q_tp1 = max(Q_tp1_all{1});
            y = r + forget_rate * Q_tp1;
        end
        Q_t_all = Qsolver.net.forward({state(:,:,:,samp_id)});
        Q_t = Q_t_all{1};
        diff_Q = single(zeros(4, 1));
        action_t = action(samp_id);
        diff_Q(action_t) = Q_t(action_t) - y;
%         if abs(diff_Q(action_t)) < 0.8;
%             samp_id = randi(im2_id-1);continue;
%         end
        Qsolver.net.backward({1*diff_Q});
        Qsolver.apply_update();
%         if samp_id == im2_id-1
        fprintf('loss = %f\n',abs(diff_Q(action_t)));
%         end
    end
%     samp_id = randi(im2_id-1);
end