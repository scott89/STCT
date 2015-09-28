function delete_solver(index)
% reset_all()
%   clear all solvers and stand-alone nets and reset Caffe to initial status

caffe_('delete_solver', index);

end
