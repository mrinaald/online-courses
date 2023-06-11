function flag = triangle_intersection(P1, P2)
% triangle_test : returns true if the triangles overlap and false otherwise

%%% All of your code should be between the two lines of stars.
% *******************************************************************

% math-only-math.com/position-of-a-point-relative-to-a-line.html

P1_lines = [
            % [(y2 - y1), (x1 - x2), (x2y1 - x1y2)]
            P1(2, 2) - P1(1, 2), P1(1, 1) - P1(2, 1), (P1(2, 1) * P1(1, 2)) - (P1(1, 1) * P1(2, 2));
            % [(y3 - y2), (x2 - x3), (x3y2 - x2y3)]
            P1(3, 2) - P1(2, 2), P1(2, 1) - P1(3, 1), (P1(3, 1) * P1(2, 2)) - (P1(2, 1) * P1(3, 2));
            % [(y1 - y3), (x3 - x1), (x1y3 - x3y1)]
            P1(1, 2) - P1(3, 2), P1(3, 1) - P1(1, 1), (P1(1, 1) * P1(3, 2)) - (P1(3, 1) * P1(1, 2));
];

P2_lines = [
            % [(y2 - y1), (x1 - x2), (x2y1 - x1y2)]
            P2(2, 2) - P2(1, 2), P2(1, 1) - P2(2, 1), (P2(2, 1) * P2(1, 2)) - (P2(1, 1) * P2(2, 2));
            % [(y3 - y2), (x2 - x3), (x3y2 - x2y3)]
            P2(3, 2) - P2(2, 2), P2(2, 1) - P2(3, 1), (P2(3, 1) * P2(2, 2)) - (P2(2, 1) * P2(3, 2));
            % [(y1 - y3), (x3 - x1), (x1y3 - x3y1)]
            P2(1, 2) - P2(3, 2), P2(3, 1) - P2(1, 1), (P2(1, 1) * P2(3, 2)) - (P2(3, 1) * P2(1, 2));
];


flag = true;

function is_parallel = check_line_parallel(line1, line2)
  if (line1(1) * line2(2)) == (line2(1) * line1(2))
    is_parallel = true;
  else
    is_parallel = false;
  end;
end

function [x, y] = get_line_intersection(line1, line2)
  a1 = line1(1);
  b1 = line1(2);
  c1 = line1(3);
  a2 = line2(1);
  b2 = line2(2);
  c2 = line2(3);

  x = ((b1 * c2) - (b2 * c1)) / ((a1 * b2) - (a2 * b1));
  y = ((c1 * a2) - (c2 * a1)) / ((a1 * b2) - (a2 * b1));
end

function is_collinear = check_collinear_points(p1, p2, p3)
  x1 = p1(1);
  y1 = p1(2);
  x2 = p2(1);
  y2 = p2(2);
  x3 = p3(1);
  y3 = p3(2);

  is_collinear = (((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2))) == 0);
end

function is_on_segment = check_point_on_segment(p1, p2, cp)
  is_collinear = check_collinear_points(p1, p2, cp);

  on_segment = (((p1(1) <= cp(1) && cp(1) <= p2(1)) || (p2(1) <= cp(1) && cp(1) <= p1(1))) && ((p1(2) <= cp(2) && cp(2) <= p2(2)) || (p2(2) <= cp(2) && cp(2) <= p1(2))));

  is_on_segment = (is_collinear && on_segment);
end

% Point-Point Intersection
  for p1=1:3
    for p2=1:3
      if isequal(P1(p1, :), P2(p2, :))
        % "point-point"
        return;
      end;
    end;
  end;

  for l1=0:2
    l11 = mod(l1, 3) + 1;
    l12 = mod(l1 + 1, 3) + 1;
    for l2=0:2
      l21 = mod(l2, 3) + 1;
      l22 = mod(l2 + 1, 3) + 1;

      if check_line_parallel(P1_lines(l11, :), P2_lines(l21, :))
        % "parallel"
        if (check_point_on_segment(P1(l11, :), P1(l12, :), P2(l21, :)) || check_point_on_segment(P1(l11, :), P1(l12, :), P2(l22, :)) || check_point_on_segment(P2(l21, :), P2(l22, :), P1(l11, :)) || check_point_on_segment(P2(l21, :), P2(l22, :), P1(l12, :)))
          % "common-line"
          return;
        end;
      else
        [x, y] = get_line_intersection(P1_lines(l11, :), P2_lines(l21, :));
        p = [x, y];
        if (check_point_on_segment(P1(l11, :), P1(l12, :), p) && check_point_on_segment(P2(l21, :), P2(l22, :), p))
          % "point-line OR line intersection"
          return;
        end;
      end;
    end;
  end;

  % P2_coords = [P2(:, :), ones(size(P2)(1), 1)]';
  % P1_side = P1_lines * P2_coords;
  % P1_side

  % P1_coords = [P1(:, :), ones(size(P1)(1), 1)]';
  % P2_side = P2_lines * P1_coords;
  % P2_side

  P1_y_lines = [
                0, 1, -P1(1, 2);
                0, 1, -P1(2, 2);
                0, 1, -P1(3, 2);
  ];

  P2_y_lines = [
                0, 1, -P2(1, 2);
                0, 1, -P2(2, 2);
                0, 1, -P2(3, 2);
  ];

  count = 0;
  for yl=0:2
    yl1 = mod(yl, 3) + 1;   % Point as well as line index
    for l21=0:2
      l22 = mod(l21 + 1, 3);        % 2nd line (0-index)
      l21p1 = mod(l21, 3) + 1;      % End-point of 1st line segment AND line index
      l21p2 = mod(l21 + 1, 3) + 1;  % End-point of 1st line segment
      l22p1 = mod(l22, 3) + 1;      % End-point of 2nd line segment AND line index
      l22p2 = mod(l22 + 1, 3) + 1;  % End-point of 2nd line segment

      if (check_line_parallel(P1_y_lines(yl1, :), P2_lines(l21p1, :)) || check_line_parallel(P1_y_lines(yl1, :), P2_lines(l22p1, :)))
        % The y-line passing through this point in P1 is parallel to some line in
        % P2. Since, all the intersection related checks are done before, this
        % pair of lines from P2 is not helpful
        continue;
      end;

      [x, y] = get_line_intersection(P1_y_lines(yl1, :), P2_lines(l21p1, :));
      p1 = [x, y];
      if ~check_point_on_segment(P2(l21p1, :), P2(l21p2, :), p1)
        % Intersection between y-line and side of P2 does not lie on P2
        continue;
      end;

      [x, y] = get_line_intersection(P1_y_lines(yl1, :), P2_lines(l22p1, :));
      p2 = [x, y];
      if ~check_point_on_segment(P2(l22p1, :), P2(l22p2, :), p2)
        % Intersection between y-line and side of P2 does not lie on P2
        continue;
      end;

      if check_point_on_segment(p1, p2, P1(yl1, :))
        count = count + 1;
        break;
      end;
    end;
  end;

  if count == 3
    % "overlapping"
    return;
  end;


  count = 0;
  for yl=0:2
    yl2 = mod(yl, 3) + 1;   % Point as well as line index
    for l11=0:2
      l12 = mod(l11 + 1, 3);        % 2nd line (0-index)
      l11p1 = mod(l11, 3) + 1;      % End-point of 1st line segment AND line index
      l11p2 = mod(l11 + 1, 3) + 1;  % End-point of 1st line segment
      l12p1 = mod(l12, 3) + 1;      % End-point of 2nd line segment AND line index
      l12p2 = mod(l12 + 1, 3) + 1;  % End-point of 2nd line segment

      if (check_line_parallel(P2_y_lines(yl2, :), P2_lines(l11p1, :)) || check_line_parallel(P2_y_lines(yl2, :), P2_lines(l12p1, :)))
        % The y-line passing through this point in P2 is parallel to some line in
        % P1. Since, all the intersection related checks are done before, this
        % pair of lines from P1 is not helpful
        continue;
      end;

      [x, y] = get_line_intersection(P2_y_lines(yl2, :), P1_lines(l11p1, :));
      p1 = [x, y];
      if ~check_point_on_segment(P1(l11p1, :), P1(l11p2, :), p1)
        % Intersection between y-line and side of P1 does not lie on P1
        continue;
      end;

      [x, y] = get_line_intersection(P2_y_lines(yl2, :), P1_lines(l12p1, :));
      p2 = [x, y];
      if ~check_point_on_segment(P1(l12p1, :), P1(l12p2, :), p2)
        % Intersection between y-line and side of P1 does not lie on P1
        continue;
      end;

      if check_point_on_segment(p1, p2, P2(yl2, :))
        count = count + 1;
        break;
      end;
    end;
  end;

  if count == 3
    % "overlapping"
    return;
  end;

flag = false;

% *******************************************************************
end
