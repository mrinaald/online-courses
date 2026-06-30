function F = EstimateFundamentalMatrix(x1, x2)
%% EstimateFundamentalMatrix
% Estimate the fundamental matrix from two image point correspondences 
% Inputs:
%     x1 - size (N x 2) matrix of points in image 1
%     x2 - size (N x 2) matrix of points in image 2, each row corresponding
%       to x1
% Output:
%    F - size (3 x 3) fundamental matrix with rank 2

N = size(x1, 1);
A = [x1(:,1).*x2(:,1), x1(:,1).*x2(:,2), x1(:,1), x1(:,2).*x2(:,1), x1(:,2).*x2(:,2), x1(:,2), x2(:,1), x2(:,2), ones(N, 1)];

[~, ~, V] = svd(A);
f = V(:, end);
F_r3 = reshape(f, 3, 3);

[U, D, V] = svd(F_r3);
D_r2 = D;
D_r2(end, end) = 0;
F_r2 = U * D_r2 * V';

F = F_r2 / norm(F_r2);




