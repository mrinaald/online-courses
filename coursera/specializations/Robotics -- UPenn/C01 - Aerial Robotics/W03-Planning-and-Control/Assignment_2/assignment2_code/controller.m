function [ u1, u2 ] = controller(~, state, des_state, params)
%CONTROLLER  Controller for the planar quadrotor
%
%   state: The current state of the robot with the following fields:
%   state.pos = [y; z], state.vel = [y_dot; z_dot], state.rot = [phi],
%   state.omega = [phi_dot]
%
%   des_state: The desired states are:
%   des_state.pos = [y; z], des_state.vel = [y_dot; z_dot], des_state.acc =
%   [y_ddot; z_ddot]
%
%   params: robot parameters

%   Using these current and desired states, you have to compute the desired
%   controls

u1 = 0;
u2 = 0;
## keyboard;

% FILL IN YOUR CODE HERE

% Using y_des to compute phi_c
k_py = 7;
k_dy = 25;
phi_c_diff = k_dy * (des_state.vel(1) - state.vel(1));
phi_c_prop = k_py * (des_state.pos(1) - state.pos(1));
phi_c = -(des_state.acc(1) + phi_c_diff + phi_c_prop) / params.gravity;


% Using phi_c to compute u_2
k_pphi = 2500;
k_dphi = 80;
u2_diff = k_dphi * ( - state.omega(1));
u2_prop = k_pphi * (phi_c - state.rot(1));
u2 = params.Ixx * (u2_diff + u2_prop);


% Using z_des to compute u_1
k_pz = 100;
k_dz = 15;
u1_diff = k_dz * (des_state.vel(2) - state.vel(2));
u1_prop = k_pz * (des_state.pos(2) - state.pos(2));
u1 = params.mass * (params.gravity + des_state.acc(2) + u1_diff + u1_prop);

end

