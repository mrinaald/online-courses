function [ predictx, predicty, state, param ] = kalmanFilter( t, x, y, state, param, previous_t )
%UNTITLED Summary of this function goes here
%   Four dimensional state: position_x, position_y, velocity_x, velocity_y

    %% Place parameters like covarainces, etc. here:
    % P = eye(4)
    % R = eye(2)

    % Check if the first time running this function
    if previous_t<0
        state = [x, y, 0, 0];
        param.P = 0.1 * eye(4);
        % param.P = 100 * eye(4);
        % param.P = diag([1e-3, 1e-3, 1e-3, 1e-3]);
        predictx = x;
        predicty = y;
        return;
    end

    dt = t - previous_t;
    z = [x; y];
    A = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1];
    C = [1 0 0 0; 0 1 0 0];
    sigma_m = diag([5e-1, 5e-1, 5e-2, 5e-2]);   % motion noise
    sigma_o = diag([5e-3, 5e-3]);               % observation noise

    %%% TODO: Add Kalman filter updates
    P = (A * param.P * A') + sigma_m;
    CPC = (C * P * C');
    R = CPC + sigma_o;

    % K = P * C' * inv(R + CPC);
    % K = (P * C') / (R + CPC);
    K = P * C' * inv(sigma_o + CPC);
    % K = (P * C') / (sigma_o + CPC);

    state_t = (A * state') + (K * (z - (C * A * state')));

    predictx = state_t(1);
    predicty = state_t(2);
    state = state_t';
    param.P = P - (K * C * P);

    %% As an example, here is a Naive estimate without a Kalman filter
    %% You should replace this code
    %vx = (x - state(1)) / (t - previous_t);
    %vy = (y - state(2)) / (t - previous_t);
    %% Predict 330ms into the future
    %predictx = x + vx * dt;
    %predicty = y + vy * dt;
    %% State is a four dimensional element
    %state = [x, y, vx, vy];
end
