
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

  % PD Controller
  Kp = 54.5;
  %proportional_gain = Kp * (0 - phi);
  proportional_gain = Kp * (phi);

  Kv = 1.3;
  %derivative_gain = Kv * (0 - phidot);
  derivative_gain = Kv * (phidot);

  phidotdot = proportional_gain + derivative_gain;

  u = phidotdot;
end

function xhatOut = EKFupdate(params, t, z)
  % z = [ay; az; gx] with a* in units of g's, and gx in units of rad/s
  % You can borrow most of your week 3 solution, but this must only implement a single predict-update step of the EKF
  % Recall (from assignment 5b) that you can use persistent variables to create/update any additional state that you need.

  % Student completes this
  %xhatOut = [0; 0];

  persistent P previousT previousX
  if isempty(P)
    % initialize
    P = eye(2, 2);
    previousT = t;
    previousX = [atan2(z(1), z(2)); z(3)];
    xhatOut = previousX;
    return
  end

  sigma_m = diag([1, 1]);                   % motion noise
  sigma_o = diag([1e-1, 1e-1, 1e-1]);       % observation noise

  dt = t - previousT;
  A = [1 dt; 0 1];
  x = previousX;
  
  Pt = (A * P * A') + sigma_m;
  H = [cos(x(1)) 0; -sin(x(1)) 0; 0 1];
  %H = [cosd(x(1)) 0; -sind(x(1)) 0; 0 1];
  %https://www.coursera.org/learn/robotics-capstone/discussions/weeks/3/threads/MWqVnH89Eea3kQ6ouhnpmw/replies/lZVJgn9iEeakbhIiKPxV8w/comments/VrvnaYDwEeaJpgoXDYAwZQ
  %H = [(pi/180)*cosd(x(1)), 0; (pi/180)*-sind(x(1)), 0; 0 1];
  K = (Pt * H') / (sigma_o + (H * Pt * H'));

  x_t = A * x;
  %h_t = [sind(x_t(1)); cosd(x_t(1)); x_t(2)];
  %h_t = [sind(x_t(1)); cosd(x_t(1)); x_t(2)] + (H * (x_t - x));
  h_t = [sin(x_t(1)); cos(x_t(1)); x_t(2)];
  xhat_t = x_t + (K * (z - h_t));
  Phat = Pt - (K * H * Pt);

  xhatOut = xhat_t;
  
  P = Phat;
  previousT = t;
  previousX = xhatOut;

end