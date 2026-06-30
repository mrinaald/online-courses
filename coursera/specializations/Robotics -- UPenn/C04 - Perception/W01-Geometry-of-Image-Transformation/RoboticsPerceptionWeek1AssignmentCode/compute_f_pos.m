 function [ f, pos ] = compute_f_pos( d1_ref, d2_ref, H1, H2, ratio, f_ref )
%% Compute camera focal length and camera position to achieve centain ratio between objects
%
% In this function, we focus on two objects: object A with height H1 and
% d1_ref as distance to camera in 3D world, object B with height H2 and
% d2_ref as distance to camera in 3D world.
% We will keep the size of object A in image the same as before while
% adjusting the size of object B in image.
%
% Input:
% - d1_ref: distance of the first object
% - d2_ref: distance of the second object
% - H1: height of the first object in physical world
% - H2: height of the second object in physical world
% - ratio: ratio between two objects in image coordinate (h1/h2)
% - f_ref: 1 by 1 double, previous camera focal length
% Output:
% - f: 1 by 1, camera focal length
% - pos: 1 by 1, camera position on z axis

% YOUR CODE HERE

% Original State
% h1_orig = (f_ref * H1) / d1_ref
% h2_orig = (f_ref * H2) / d2_ref

% After changing to new position
% h1 = (f * H1) / (d1_ref - pos)
% h2 = (f * H2) / (d2_ref - pos)

% We want that h1 == h1_orig, which implies
% f = ((d1_ref - pos) * f_ref) / d1_ref

% We are also provided h1/h2 ratio, which implies
% ratio == (H1 / (d1_ref - pos)) * ((d2_res - pos) / H2), which implies
% pos = (H2 * d1_ref * ratio - h1 * d2_ref) / (H2 * ratio - H1)

pos = ((H2 * d1_ref * ratio) - (H1 * d2_ref)) / ((H2 * ratio) - H1);

f = ((d1_ref - pos) * f_ref) / d1_ref;

end

