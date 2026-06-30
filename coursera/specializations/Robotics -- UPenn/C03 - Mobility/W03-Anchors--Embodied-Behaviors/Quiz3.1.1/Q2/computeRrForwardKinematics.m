function [elbow,endeff] = computeRrForwardKinematics(rads1,rads2)
%%GIVEN THE ANGLES OF THE MOTORS, return an array of arrays for the
%%position of the elbow [x1,y1], and endeffector [x2,y2]

%rads1 = 1;
%rads2 = 0.75;

x1 = cos(rads1);
y1 = sin(rads1);

x2 = x1 + cos(rads1 + rads2);
y2 = y1 + sin(rads1 + rads2);

elbow = [x1, y1];
endeff =[x2, y2];
