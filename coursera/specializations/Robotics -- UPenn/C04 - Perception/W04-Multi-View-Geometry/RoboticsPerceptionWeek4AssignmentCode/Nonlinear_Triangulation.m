function X = Nonlinear_Triangulation(K, C1, R1, C2, R2, C3, R3, x1, x2, x3, X0)
%% Nonlinear_Triangulation
% Refining the poses of the cameras to get a better estimate of the points
% 3D position
% Inputs: 
%     K - size (3 x 3) camera calibration (intrinsics) matrix
%     x
% Outputs: 
%     X - size (N x 3) matrix of refined point 3D locations 
N = size(X0, 1);
X = zeros(N, 3);

for i=1:N
   X_new = Single_Point_Nonlinear_Triangulation(K, C1, R1, C2, R2, C3, R3, x1(i, :)', x2(i, :)', x3(i, :)', X0(i, :)');
   X(i, :) = X_new;
end
end

function X = Single_Point_Nonlinear_Triangulation(K, C1, R1, C2, R2, C3, R3, x1, x2, x3, X0)
image_pts1 = K * R1 * (X0 - C1);
image_pts1 = image_pts1 ./ image_pts1(3);
image_pts2 = K * R2 * (X0 - C2);
image_pts2 = image_pts2 ./ image_pts2(3);
image_pts3 = K * R3 * (X0 - C3);
image_pts3 = image_pts3 ./ image_pts3(3);

b = [x1; x2; x3];
f_X = [image_pts1(1) image_pts1(2) image_pts2(1) image_pts2(2) image_pts3(1) image_pts3(2)]';

J1 = Jacobian_Triangulation(C1, R1, K, X0);
J2 = Jacobian_Triangulation(C2, R2, K, X0);
J3 = Jacobian_Triangulation(C3, R3, K, X0);
J = [J1' J2' J3']';

delta_X = (J' * J) \ (J' * (b - f_X));
X = X0 + delta_X;
end

function J = Jacobian_Triangulation(C, R, K, X)
image_pts = K * R * (X - C);
u = image_pts(1);
v = image_pts(2);
w = image_pts(3);

f = K(1, 1);
px = K(1, 3);
py = K(2, 3);

du_by_dX = [f * R(1, 1) + px * R(3, 1),  f * R(1, 2) + px * R(3, 2), f * R(1, 3) + px * R(3, 3)];
dv_by_dX = [f * R(2, 1) + py * R(3, 1),  f * R(2, 2) + py * R(3, 2), f * R(2, 3) + py * R(3, 3)];
dw_by_dX = [R(3, 1) R(3, 2) R(3, 3)];

df_by_dX = [w * du_by_dX - u * dw_by_dX; w * dv_by_dX - v * dw_by_dX];
J = df_by_dX / (w * w);
end
