function [rads1,rads2] = computeRrInverseKinematics(X,Y)

% x=0, y=1.5; 0 < \theta_θ1 ​< \frac{\pi}{2}, 0< \theta_θ2 ​< 2\pi
%X = 0;
%Y = 1.5;

syms theta1 theta2 ;
eqns = [
        theta1 > 0, ...
        theta1 < (pi / 2), ...
        theta2 > 0, ...
        theta2 < (2 * pi), ...
        cos(theta1) + cos(theta1 + theta2) == X, ...
        sin(theta1) + sin(theta1 + theta2) == Y
];
vars = [theta1, theta2];

%rads1=0;
%rads2=0;
[rads1, rads2] = solve(eqns, vars);

%A = [X, Y, rads1, rads2]
