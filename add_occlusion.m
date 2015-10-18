function [fea_occ] = add_occlusion(varargin)
fea = varargin{1};
[h, w, ~] = size(fea);
if nargin == 1
    half_w = floor(w/2);
    half_h = floor(h/2);
    
    fea_occ = repmat(fea, 1, 1, 1, 5);
    fea_occ(1:half_h, :, :, 2) =  0;
    fea_occ(half_h:h, :, :, 3) =  0;
    fea_occ(:, 1:half_w, :, 4) = 0;
    fea_occ(:, half_w:w, :, 5) = 0;
elseif nargin == 2
    center = varargin{2};
    fea_occ = repmat(fea, 1, 1, 1, 5);
    fea_occ(1:center(2), :, :, 2) =  0;
    fea_occ(center(2):h, :, :, 3) =  0;
    fea_occ(:, 1:center(1), :, 4) = 0;
    fea_occ(:, center(1):w, :, 5) = 0;
end