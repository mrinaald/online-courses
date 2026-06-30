function [endeff] = computeMiniForwardKinematics(rads1,rads2)

% \theta1 ​= 3.5, \theta2 = 1.5
%rads1 = 3.5;
%rads2 = 1.5;

alpha = pi + ((rads1 + rads2) / 2);
beta = (rads1 - rads2) / 2;

r = (2 * cos(beta / 2)) - cos(beta);
x = r * cos(alpha);
y = r * sin(alpha);

endeff = [x, y];
