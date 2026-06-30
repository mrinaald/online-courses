function Tphi = pitchController(phi,phiDesired,dphi_dt)
kd_phi = 5;
kp_phi = 5;

proportional_gain = kp_phi * (phiDesired - phi);
derivative_gain = kd_phi * (0 - dphi_dt);
Tphi = proportional_gain + derivative_gain;