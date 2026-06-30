
function u = controller(params, t, X)
  u=[0; 0];
  % 1. write out the forward kinematics, such that p = FK(theta1, theta2)
  % 2. Let e = p - params.traj(t) be the task-space error
  % 3. Calculate the manipulator Jacobian J = d p / d theta
  % 4. Use a "natural motion" PD controller, u = - kp * J^T * e - kd * [dth1; dth2]
end

