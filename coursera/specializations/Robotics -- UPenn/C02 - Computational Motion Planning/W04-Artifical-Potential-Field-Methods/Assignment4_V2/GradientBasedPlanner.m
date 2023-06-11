function route = GradientBasedPlanner (f, start_coords, end_coords, max_its)
% GradientBasedPlanner : This function plans a path through a 2D
% environment from a start to a destination based on the gradient of the
% function f which is passed in as a 2D array. The two arguments
% start_coords and end_coords denote the coordinates of the start and end
% positions respectively in the array while max_its indicates an upper
% bound on the number of iterations that the system can use before giving
% up.
% The output, route, is an array with 2 columns and n rows where the rows
% correspond to the coordinates of the robot as it moves along the route.
% The first column corresponds to the x coordinate and the second to the y coordinate

[gx, gy] = gradient (-f);

%%% All of your code should be between the two lines of stars.
% *******************************************************************
%The section of code that you are asked to complete should perform the following procedure:
%- On every iteration the planner should update the position of the robot based on the gradient
%  values contained in the arrays gx and gy. Make sure you normalize the gradient vectors.
%- Update the route by adding the new position of the robot to the end of the route array.
%  Note that the distance between successive locations in the route should not be greater than 1.0.
%- Continue the same procedure until the distance between the robot’s current position and the goal
%  is less than 2.0 or the number of iterations exceeds the value contained in max_its.

function distance = compute_distance(x, y)
  distance = norm(y - x);
end

[nrows, ncols] = size(f);

route = [start_coords];

for iter=1:max_its-1
  current = route(end, :);
  x_pos = round(current(1));
  y_pos = round(current(2));

  % f(current(1)-4:current(1)+4, current(2)-4:current(2)+4)
  % gx(current(1)-4:current(1)+4, current(2)-4:current(2)+4)
  % gy(current(1)-4:current(1)+4, current(2)-4:current(2)+4)
  % break;

  % v = [gx(x_pos, y_pos), gy(x_pos, y_pos)];
  v = [gx(y_pos, x_pos), gy(y_pos, x_pos)];

  v = v / norm(v);
  next = current + v;

  % if abs(v(1)) >= abs(v(2))
  %   if (v(1) > 0) && (current(1) + 1 <= nrows)
  %     next = current + [1, 0];
  %   elseif (v(1) < 0) && (current(1) - 1 >= 1)
  %     next = current + [-1, 0];
  %   else
  %     next = current;
  %   end
  % else
  %   if (v(2) > 0) && (current(2) + 1 <= ncols)
  %     next = current + [0, 1];
  %   elseif (v(2) < 0) && (current(2) - 1 >= 1)
  %     next = current + [0, -1];
  %   else
  %     next = current;
  %   end
  % end

  route = [route; next];
  if compute_distance(next, end_coords) < 2.0
    break;
  end
end

% route = [route; 137.5, 275; end_coords]
% route = [route; end_coords];
route = double(route);
size(route)

% *******************************************************************
end
