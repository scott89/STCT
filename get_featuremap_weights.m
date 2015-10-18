function w0 = get_featuremap_weights(weight_dim, num_occlusion)
w0 = single(zeros(1, 1, weight_dim, num_occlusion));

if mod(weight_dim, num_occlusion) ~= 0
    error('mod(weight_dim, num_occlusion) ~= 0');
end
start = 1;
stride = weight_dim / num_occlusion;
for i = 1:num_occlusion
    w0(1, 1, start:start+stride-1, i) = 1;
    start = start + stride;
end
