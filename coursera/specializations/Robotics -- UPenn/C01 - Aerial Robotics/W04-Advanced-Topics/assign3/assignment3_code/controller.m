function [F, M] = controller(t, state, des_state, params)
%CONTROLLER  Controller for the quadrotor
%
%   state: The current state of the robot with the following fields:
%   state.pos = [x; y; z], state.vel = [x_dot; y_dot; z_dot],
%   state.rot = [phi; theta; psi], state.omega = [p; q; r]
%
%   des_state: The desired states are:
%   des_state.pos = [x; y; z], des_state.vel = [x_dot; y_dot; z_dot],
%   des_state.acc = [x_ddot; y_ddot; z_ddot], des_state.yaw,
%   des_state.yawdot
%
%   params: robot parameters

%   Using these current and desired states, you have to compute the desired
%   controls


% =================== Your code goes here ===================
persistent mrinaal

% Thrust
F = 0;

% Moment
M = zeros(3,1);

% k_px, k_py, k_pz ~ 100 - 1000
% k_dx, k_dy, k_dz ~ 0 - 10
## k_pr = [50, 50, 50];   % works for line
## k_dr = [5, 5, 5];
k_pr = [50, 50, 50];   % works for line
k_dr = [5, 5, 5];

r_ddot_c = zeros(3, 1);
for i=1:3
  diff_gain = k_dr(i) * (des_state.vel(i) - state.vel(i));
  prop_gain = k_pr(i) * (des_state.pos(i) - state.pos(i));
  r_ddot_c(i) = des_state.acc(i) + diff_gain + prop_gain;
end

% Commanding Thrust
F = params.mass * (params.gravity + r_ddot_c(3));


phi_c = ((r_ddot_c(1) * sin(des_state.yaw)) - (r_ddot_c(2) * cos(des_state.yaw))) / params.gravity;
theta_c = ((r_ddot_c(1) * cos(des_state.yaw)) + (r_ddot_c(2) * sin(des_state.yaw))) / params.gravity;

% Commanding Moment
% k_pphi, k_ptheta, ~ 1000 - 10000
% k_dphi, k_dtheta ~ 10 - 1000
% k_ppsi, k_dpsi ~ nominal
## k_pa = [2500, 2500, 10];  % works for line
## k_da = [80, 80, 1];
k_pa = [2500, 2500, 10];
k_da = [80, 80, 1];
attitude_c = [phi_c, theta_c, des_state.yaw];
attitude_vel = [0, 0, des_state.yawdot];
for i=1:3
  diff_gain = k_da(i) * (attitude_vel(i) - state.omega(i));
  prop_gain = k_pa(i) * (attitude_c(i) - state.rot(i));
  M(i) = diff_gain + prop_gain;
end
M = params.I * M;


% =================== Your code ends here ===================

end
