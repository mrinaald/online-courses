function [ u ] = pd_controller(~, s, s_des, params)
%PD_CONTROLLER  PD controller for the height
%
%   s: 2x1 vector containing the current state [z; v_z]
%   s_des: 2x1 vector containing desired state [z; v_z]
%   params: robot parameters

u = 0;


% FILL IN YOUR CODE HERE
Kp = 70;
proportional_gain = Kp * (s_des(1) - s(1));
Ke = 10;
derivative_gain = Ke * (s_des(2) - s(2));
u = params.mass * (params.gravity + proportional_gain + derivative_gain);

end

