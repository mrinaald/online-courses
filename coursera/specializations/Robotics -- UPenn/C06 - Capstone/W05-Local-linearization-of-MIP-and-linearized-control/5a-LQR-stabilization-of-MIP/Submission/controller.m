
function u = controller(params, t, X)
  % You have full state feedback available
  u=0;

  % After doing the steps in simLinearization, you should be able to substitute the linear controller u = -K*x
  K = [-1.0000 -113.1952   -1.2465  -13.9340];
  u = -K * X;
end
