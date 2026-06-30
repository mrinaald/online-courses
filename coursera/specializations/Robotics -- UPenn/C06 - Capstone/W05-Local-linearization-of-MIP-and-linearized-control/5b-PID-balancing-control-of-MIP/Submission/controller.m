
function u = controller(params, t, phi, phidot)
  % STUDENT FILLS THIS OUT
  % 
  % Initialize any added state like this:
  % 
  % persistent newstate
  % if isempty(newstate)
  %   % initialize
  %   newstate = 0;
  % end
  % 

  %%{
  % PD Controller
  Kp = 54.5;
  %proportional_gain = Kp * (0 - phi);
  proportional_gain = Kp * (phi);

  Kv = 1.3;
  %derivative_gain = Kv * (0 - phidot);
  derivative_gain = Kv * (phidot);

  phidotdot = proportional_gain + derivative_gain;

  u = phidotdot;
  %%}

  %% https://www.coursera.org/learn/robotics-capstone/discussions/forums/XWgRGjJcEeaAgxK0X1Pyiw/threads/6Bo2MD7qEeaS0w6RXgoWAw
  %{
  persistent totalError previousT
  if isempty(totalError)
      totalError = 0;
      previousT = 0;
  end
  
  Kp = 49.2;
  %proportional_gain = Kp * (0 - phi);
  proportional_gain = Kp * (phi);

  Kv = 1.7;
  %derivative_gain = Kv * (0 - phidot);
  derivative_gain = Kv * (phidot);

  dt = t - previousT;
  totalError = totalError + abs(phi) + abs(phidot * dt); % adding error in position and velocity
  %totalError = totalError + (phidot * dt); % adding error in position and velocity
  Ki = -2000;
  integral_gain = Ki * totalError;

  %alpha = (params.r + params.d) / (-1 * params.r * params.ir);

  phidotdot = proportional_gain + derivative_gain + integral_gain;

  %u = phidotdot / alpha;
  u = phidotdot;
  %}
end
