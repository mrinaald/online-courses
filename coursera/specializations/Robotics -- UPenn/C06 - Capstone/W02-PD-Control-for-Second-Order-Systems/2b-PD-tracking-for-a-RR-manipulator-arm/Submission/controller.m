
function u = controller(params, t, X)
  u=[0; 0];
  % 1. write out the forward kinematics, such that p = FK(theta1, theta2)
  % 2. Let e = p - params.traj(t) be the task-space error
  % 3. Calculate the manipulator Jacobian J = d p / d theta
  % 4. Use a "natural motion" PD controller, u = - kp * J^T * e - kd * [dth1; dth2]

  p = params.l * [cos(X(1)) + cos(X(1) + X(2)); sin(X(1)) + sin(X(1) + X(2))];

  J = [-p(2), -params.l * sin(X(1) + X(2)); p(1), params.l*cos(X(1) + X(2))];

  Kp = 20000;
  proportional_gain = -Kp * J' * (p - params.traj(t));

  Kd = 100;
  derivative_gain = -Kd * [X(3); X(4)];

  u = proportional_gain + derivative_gain;
end
