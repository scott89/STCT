function [ A ] = rotate_mat(alpha)
A = [cos(alpha), -sin(alpha);
    sin(alpha), cos(alpha)];
end

