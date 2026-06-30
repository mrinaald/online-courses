
function xhat = EKFstudent(t, z)
  % In this exercise, you will batch-process this data: you are provided a vector of timestamps (of length T), and a 3xT matrix of observations, z.
  xhat = zeros(2,length(t));

  % Student completes this
  % Initial condtion
  P = eye(2, 2);
  x_0 = [atand(z(1, 1) / z(2, 1)); z(3, 1)];
  xhat(:, 1) = x_0;
  N = length(t);

  %% err = [690.4444  616.9450]
  %sigma_m = diag([1, 1e-2]);             % motion noise
  %sigma_o = diag([1e-2, 1e-2, 1]);       % observation noise

  %% err = [668.7536  589.5858]
  %sigma_m = diag([5e-1, 5e-1]);             % motion noise
  %sigma_o = diag([1e-3, 1e-3, 15]);       % observation noise
  
  
  sigma_m = diag([1, 5]);             % motion noise
  sigma_o = diag([5e-3, 5e-3, 10]);       % observation noise

  for ti=2:N
      dt = t(ti) - t(ti-1);
      A = [1 dt; 0 1];
      x = xhat(:, ti-1);
      
      Pt = (A * P * A') + sigma_m;
      H = [cosd(x(1)) 0; -sind(x(1)) 0; 0 1];
      K = (Pt * H') / (sigma_o + (H * Pt * H'));

      x_t = A * x;
      h_t = [sind(x_t(1)); cosd(x_t(1)); x_t(2)];
      xhat_t = x_t + (K * (z(:, ti) - h_t));
      Phat = Pt - (K * H * Pt);

      xhat(:, ti) = xhat_t;
      P = Phat;
  end
end