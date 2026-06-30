function [C, R] = LinearPnP(X, x, K)
%% LinearPnP
% Getting pose from 2D-3D correspondences
% Inputs:
%     X - size (N x 3) matrix of 3D points
%     x - size (N x 2) matrix of 2D points whose rows correspond with X
%     K - size (3 x 3) camera calibration (intrinsics) matrix
% Outputs:
%     C - size (3 x 1) pose transation
%     R - size (3 x 1) pose rotation
%
% IMPORTANT NOTE: While theoretically you can use the x directly when solving
% for the P = [R t] matrix then use the K matrix to correct the error, this is
% more numeically unstable, and thus it is better to calibrate the x values
% before the computation of P then extract R and t directly

N = size(x, 1);
x_h = [x ones(N, 1)];

x_c = K \ x_h';    % inv(K) * x_h' is slower and less accurate than this
x_c = x_c';

function a = get_constants(image_pts, world_pts)
    % R = [r11, r12, r13; r21, r22, r23; r31, r32, r33]
    % t = [t1; t2; t3]
    % [x; y; 1] = [R t] * [X; Y; Z; 1]
    % -X.r11  0.r21 + x.X.r31 + -Y.r12 +  0.r22 + x.Y.r32 + -Z.r13 +  0.r23 + x.Z.r33 +  -t1 + 0.t2 + x.t3 = 0
    %  0.r11 -X.r21 + y.X.r31 +  0.r12 + -Y.r22 + y.Y.r32 +  0.r13 + -Z.r23 + y.Z.r33 + 0.t1 +  -t2 + y.t3 = 0
    
    xi = image_pts(1) / image_pts(3);
    yi = image_pts(2) / image_pts(3);
    Xw = world_pts(1);
    Yw = world_pts(2);
    Zw = world_pts(3);
    a1 = [...
        -Xw, 0, xi * Xw, ...
        -Yw, 0, xi * Yw, ...
        -Zw, 0, xi * Zw, ...
         -1, 0,      xi, ...
         ];
    a2 = [...
        0, -Xw, yi * Xw, ...
        0, -Yw, yi * Yw, ...
        0, -Zw, yi * Zw, ...
        0,  -1,      yi, ...
         ];
    a = [a1; a2];
end

A = zeros(2 * N ,12);
for i=1:N
    a = get_constants(x_c(i, :), X(i, :));
    % A = [A; a];
    A(2*i - 1, :) = a(1, :);
    A(2*i, :) = a(2, :);
end

[~, ~, V] = svd(A);

p = V(:, end);
P = reshape(p, 3, 4);

R_prime = P(:, 1:3);
t_prime = P(:, 4);

[U, D, V] = svd(R_prime);
d = det(U * V');

if d < 0
    R = -U * V';
    t = -t_prime / D(1, 1);
    C = -R' * t;
elseif d > 0
    R = U * V';
    t = t_prime / D(1, 1);
    C = R' * t;
end

end