function [ H ] = est_homography(video_pts, logo_pts)
% est_homography estimates the homography to transform each of the
% video_pts into the logo_pts
% Inputs:
%     video_pts: a 4x2 matrix of corner points in the video
%     logo_pts: a 4x2 matrix of logo points that correspond to video_pts
% Outputs:
%     H: a 3x3 homography matrix such that logo_pts ~ H*video_pts
% Written for the University of Pennsylvania's Robotics:Perception course

% YOUR CODE HERE
H = [];

    function [ a ] = get_homography_equation_constants(src_pts, dest_pts)
        a_x = [...
                         -src_pts(1),              -src_pts(2),         -1,...
                                   0,                        0,          0,...
            dest_pts(1) * src_pts(1), dest_pts(1) * src_pts(2), dest_pts(1)...
            ];
        a_y = [...
                                   0,                        0,          0,...
                         -src_pts(1),              -src_pts(2),         -1,...                                   
            dest_pts(2) * src_pts(1), dest_pts(2) * src_pts(2), dest_pts(2)...
            ];
        a = [a_x; a_y];
    end

a1 = get_homography_equation_constants(video_pts(1, :), logo_pts(1, :));
a2 = get_homography_equation_constants(video_pts(2, :), logo_pts(2, :));
a3 = get_homography_equation_constants(video_pts(3, :), logo_pts(3, :));
a4 = get_homography_equation_constants(video_pts(4, :), logo_pts(4, :));

A = [a1; a2; a3; a4];

[~, ~, V] = svd(A);

h = V(:, end);

H = reshape(h, 3, 3);
H = H';

end

