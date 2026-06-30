
function u = controller(params, t, x, xd)
  % x = current position
  % xd = current velocity

  % Use params.traj(t) to get the reference trajectory
  % e.g. (x - params.traj(t)) represents the instaneous trajectory error

  % params can be initialized in the initParams function, which is called before the simulation starts
  
  % SOLUTION GOES HERE -------------

  % This way of tuning could also be tried.
  % Reference: https://www.coursera.org/learn/robotics-capstone/discussions/weeks/2/threads/1T5QGz1EEeaXnBKVQldqyw/replies/dFpzNT3SEeaDRA5SxbW7qQ/comments/N-btfj3YEeaS0w6RXgoWAw
  %f = 100;
  %z = 5;
  %Kp = (2*pi*f)^2; 
  %Kd =  4*pi*f*z;

  Kp = 4000;
  proportional_gain = Kp * (params.traj(t) - x);

  Kv = 1000;
  derivative_gain = Kv * (-xd);

  u = proportional_gain + derivative_gain;
end