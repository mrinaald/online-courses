
function u = controllerNoisyEnc(params, t, obs, th, dth)
  % This is the starter file for the week5 assignment
  % Now you only receive noisy measurements for theta, and must use your EKF from week 3 to filter the data and get an estimate of the state
  % obs = [ay; az; gx] (same as last week)
  % New for 6b: you also have access to params.traj(t)

   persistent counter
   if t == 0
       counter = 0;
   end

  % Template code (same as last week)
  xhat = EKFupdate(params, t, obs);
  phi = xhat(1);
  phidot = xhat(2);

  % fill this out
  %u=0;
  %% TODO: read this:
  %% https://www.coursera.org/learn/robotics-capstone/discussions/forums/0jAAfjJcEea56g4WAX2f2Q/threads/MjklBa7XEeyE-RItVp0rOw

  % Calculating phi_des
  x = params.r * (th + phi);
  xdot = params.r * (dth + phidot);
  %kpx = 0.2;
  %kpx = 0.2;
  kpx = 0.15;       % works for step
  %kpx = 0.1;
  prop_gain_x = kpx * (params.traj(t) - x);

  %kdx = 0.015;
  %kdx = 0.03;
  kdx = 0.022;       % works for step
  %kdx = 0.04;
  deriv_gain_x = kdx * (0 - xdot);

  phi_des = prop_gain_x + deriv_gain_x;

  
  
  % PD Controller for u
  %Kp = 54.5;
  %Kp = 47;
  %Kp = 52;
  Kp = 45;       % works for step
  %Kp = 45;
  %proportional_gain = Kp * (0 - phi);
  proportional_gain = Kp * sin(phi - phi_des);

  %Kv = 1.3;
  %Kv = 0.3;
  %Kv = 0.19;
  Kv = 0.1;       % works for step
  %Kv = 0.15;
  %derivative_gain = Kv * (0 - phidot);
  derivative_gain = Kv * (phidot);

  phidotdot = proportional_gain + derivative_gain;
  u = phidotdot;

  %{
  if counter == 0
      fprintf('params.traj(t) = %.4f\t th = %.4f\t phi = %.4f\t x = %.4f\n', params.traj(t), th, phi, x)
  end
  if mod(counter, 5000) == 0
      fprintf('t = %.2fs\t ',t)
      %fprintf('prop_gain_x = %.4f\t deriv_gain_x = %.4f\t ', prop_gain_x, deriv_gain_x)
      %fprintf('prop_gain_u = %.4f\t deriv_gain_u = %.4f\t ', proportional_gain, derivative_gain)
      fprintf('prop_gain_x = %.4f\t deriv_gain_x = %.4f\t ', params.traj(t) - x, 0 - xdot)
      fprintf('prop_gain_u = %.4f\t deriv_gain_u = %.4f\t ', sin(phi - phi_des), phidot)
      fprintf('u = %.4f\n', u)
  end
  counter = counter + 1;
  %}
end

function xhatOut = EKFupdate(params, t, z)
  % z = [ay; az; gx] with a* in units of g's, and gx in units of rad/s
  % You can borrow most of your week 3 solution, but this must only implement a single predict-update step of the EKF
  % Recall (from assignment 5b) that you can use persistent variables to create/update any additional state that you need.

  % Student completes this
  %xhatOut = [0; 0];

  persistent P previousT previousX
  if t == 0
    % initialize
    P = eye(2, 2);
    previousT = t;
    %fprintf('sin(phi) = %.4f\t cos(phi) = %.4f\t phi = %.4f\n', z(1), z(2), atan2(z(1), z(2)))
    previousX = [atan2(z(1), z(2)); z(3)];
    xhatOut = previousX;
    return
  end

  %sigma_m = diag([1, 1]);                   % motion noise
  %sigma_o = diag([1e-1, 1e-1, 1e-1]);       % observation noise
  %sigma_m = diag([1.5, 1.5]);                   % motion noise
  %sigma_o = diag([7e-2, 7e-2, 7e-2]);       % observation noise
  sigma_m = diag([2.5, 2.5]);                   % motion noise
  sigma_o = diag([1e-3, 1e-3, 1e-3]);       % observation noise

  dt = t - previousT;
  A = [1 dt; 0 1];
  x = previousX;
  
  Pt = (A * P * A') + sigma_m;
  H = [cos(x(1)) 0; -sin(x(1)) 0; 0 1];
  %H = [cosd(x(1)) 0; -sind(x(1)) 0; 0 1];
  %https://www.coursera.org/learn/robotics-capstone/discussions/weeks/3/threads/MWqVnH89Eea3kQ6ouhnpmw/replies/lZVJgn9iEeakbhIiKPxV8w/comments/VrvnaYDwEeaJpgoXDYAwZQ
  %H = [(pi/180)*cosd(x(1)), 0; (pi/180)*-sind(x(1)), 0; 0 1];
  hph = (sigma_o + (H * Pt * H'));
  if isnan(rcond(hph))
      error('Matrix is singular, close to singular or badly scaled. Results may be inaccurate. RCOND = NaN.')
  end
  K = (Pt * H') / hph;

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