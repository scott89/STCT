function [ overlap ] = ComputeOverlap(B1, B2)
% input: stack of bounding boxes B1 and B2: [ltx, lty, w, h; ...]
% output overlaps between B1 and B2
xt1 = B1(:, 1); yt1 = B1(:, 2); w1 = B1(:, 3); h1 = B1(:, 4);
xb1 = xt1 + w1 - 1; yb1 = yt1 + h1 - 1;
% loc1 = [xt1, yt1, xb1, yb1];
%
xt2 = B2(:, 1); yt2 = B2(:, 2); w2 = B2(:, 3); h2 = B2(:, 4);
xb2 = xt2 + w2 - 1; yb2 = yt2 + h2 - 1;
% loc2 = [xt2, yt2, xb2, yb2];
%
x1 = max(xt1, xt2);
y1 = max(yt1, yt2);

x2 = min(xb1, xb2);
y2 = min(yb1, yb2);

a = x2 - x1;
a = a.*double(a>0);
b = y2 - y1;
b = b.*double(b>0);

I = a.*b;
U = w1.*h1 + w2.*h2 -  I;
overlap = I./U;
end

