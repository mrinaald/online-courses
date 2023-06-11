% Point-Point intersection
% x = [2,2,3];
% y = [5,3,4];
% P1 = [x;y]';

% x = [0,4,2];
% y = [0,4,5];
% P2 = [x;y]';

% Point-line Intersection
% x = [1,3,5];
% y = [1,5,1];
% P1 = [x;y]';

% x = [4,6,8];
% y = [3,7,8];
% P2 = [x;y]';

% Common line Intersection
% x = [1,2,5];
% y = [3,5,3];
% P1 = [x;y]';

% x = [2,4,3];
% y = [3,3,1];
% P2 = [x;y]';

% Standard Intersection
% x = [1,1,5];
% y = [1,5,1];
% P1 = [x;y]';

% x = [6,8,3];
% y = [3,3,2];
% P2 = [x;y]';

% Overlapping
x = [2,2,3];
y = [5,3,4];
P1 = [x;y]';

x = [0,4,2];
y = [0,4,6];
P2 = [x;y]';

% No intersection
% x = [1,4,4];
% y = [1,1,3];
% P1 = [x;y]';

% x = [4,2,3];
% y = [4,8,8];
% P2 = [x;y]';

% Official Test Case
% x = [1,2,1];
% y = [1,2,3];
% P1 = [x;y]';

% x = [3,4,4];
% y = [4,2,4];
% P2 = [x;y]';


line([P1(:,1)' P1(1,1)],[P1(:,2)' P1(1,2)],'Color','r')
line([P2(:,1)' P2(1,1)],[P2(:,2)' P2(1,2)],'Color','b')

flag = triangle_intersection(P1,P2);
if flag == 1
  disp("Intersection")
else
  disp("No Intersection")
end
