function qdd = eom(params, th, phi, dth, dphi, u)
  % This is the starter file for the week5 assignment

  % Provided params are
  % params.g: gravitational constant
  % params.mr: mass of the "rod"
  % params.ir: rotational inertia of the rod
  % params.d: distance of rod CoM from the wheel axis
  % params.r: wheel radius

  % Provided states are:
  % th: wheel angle (relative to body)
  % phi: body pitch
  % dth, dphi: time-derivatives of above
  % u: torque applied at the wheel

  qdd = [0;0];
  % THE STUDENT WILL FILL THIS OUT

  %% My Original working solution
  %phidd_den = params.mr * params.d * params.d * sin(phi) * sin(phi);
  %phidd_den = phidd_den + params.ir;
  %phidd_num = -u;
  %phidd_num = phidd_num + ((-u * params.d * cos(phi)) / params.r);
  %phidd_num = phidd_num + (params.mr * params.d * sin(phi) * (params.g - (params.d * dphi * dphi * cos(phi))));

  %phidd = phidd_num / phidd_den;

  %thetadd_den = params.mr * params.r * params.r;
  %thetadd_num = (params.mr * params.r * params.d * dphi * dphi * sin(phi));
  %thetadd_num = thetadd_num + u;
  %thetadd_num = thetadd_num - (phidd * params.mr * params.r * (params.r + (params.d * cos(phi))));

  %thetadd = thetadd_num / thetadd_den;

  %qdd = [thetadd; phidd];

  %% Solution hint taken from
  %% https://www.coursera.org/learn/robotics-capstone/discussions/weeks/4/threads/hvdM8zRzEeyj8g6I-69u0w/replies/_EMhq1CgEey41RIHax-r7Q
  mr2 = params.mr * params.r * params.r;
  mrlcosphi = params.mr * params.r * params.d * cos(phi);
  mrlsinphi = params.mr * params.r * params.d * sin(phi);
  A = [mr2, mr2 + mrlcosphi;...
       mr2 + mrlcosphi, mr2 + (params.mr * params.d * params.d) + (2 * mrlcosphi) + params.ir];

  B = [u + (mrlsinphi * dphi * dphi); params.mr * params.d * sin(phi) * (params.g + (params.r * dphi * dphi))];

  qdd = A \ B;

end