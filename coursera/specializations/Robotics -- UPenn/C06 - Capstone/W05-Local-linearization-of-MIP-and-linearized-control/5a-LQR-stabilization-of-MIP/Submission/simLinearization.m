
params = struct();

params.g = 9.81;
params.mr = 0.25;
params.ir = 0.0001;
params.d = 0.1;
params.r = 0.02;

% 1. Learn how to use the symbolic toolbox in MATLAB
% you will need the state variables, and control input to be declared as symbolic
syms phi dphi th dth u;

% 2. call your "eom" function to get \ddot{q} symbolically
qdd = simplify(eom(params, th, phi, dth, dphi, u));

% 3. Linearize the system at 0 (as shown in lecture)
% You should end up with A (4x4), and b (4x1)
x = [th phi dth dphi];
f = [dth dphi qdd(1) qdd(2)];
symA = jacobian(f, x);
A = double(subs(symA, [th phi dth dphi], [0 0 0 0]));
symb = jacobian(f, u);
b = double(subs(symb, [th phi dth dphi], [0 0 0 0]));

% 4. Check that (A,b) is  controllable
% Number of uncontrollable states should return 0
co = ctrb(A, b);
unco = length(A) - rank(co);

% 5. Use LQR to get K as shown in the lecture
K = lqr(A, b, eye(4), 1, 0)