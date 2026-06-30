
function u = controllerNoisy(params, t, obs)
  % This is the starter file for the week5 assignment
  % Now you only receive noisy measurements for phi, and must use your EKF from week 3 to filter the data and get an estimate of the state
  % obs = [ay; az; gx] with a* in units of g's, and gx in units of rad/s

  % This template code calls the function EKFupdate that you must complete below
  xhat = EKFupdate(params, t, obs);
  phi = xhat(1);
  phidot = xhat(2);

  % The rest of this function should ideally be identical to your solution in week 4
  % Student completes this
  u = 0;
end

function xhatOut = EKFupdate(params, t, z)
  % z = [ay; az; gx] with a* in units of g's, and gx in units of rad/s
  % You can borrow most of your week 3 solution, but this must only implement a single predict-update step of the EKF
  % Recall (from assignment 5b) that you can use persistent variables to create/update any additional state that you need.

  % Student completes this
  xhatOut = [0; 0];
end
