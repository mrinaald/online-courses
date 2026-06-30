% Robotics: Estimation and Learning 
% WEEK 4
% 
% Complete this function following the instruction. 
function myPose = particleLocalization_bkp(ranges, scanAngles, map, param)

% Number of poses to calculate
N = size(ranges, 2);
% Output format is [x1 x2, ...; y1, y2, ...; z1, z2, ...]
myPose = zeros(3, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Map Parameters 
% 
% % the number of grids for 1 meter.
myResolution = param.resol;
% % the origin of the map in pixels
myOrigin = param.origin; 

% The initial pose is given
myPose(:,1) = param.init_pose;
% You should put the given initial pose into myPose for j=1, ignoring the j=1 ranges. 
% The pose(:,1) should be the pose when ranges(:,j) were measured.

numRows = size(map, 1);
numCols = size(map, 2);


% Decide the number of particles, M.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = 200;                           % Please decide a reasonable number of M, 
                               % based on your experiment using the practice data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create M number of particles
P = repmat(myPose(:,1), [1, M]);

std = [0.07; 0.07; 5 * (pi / 180)];
W = ones(1, M) / M;

scanAnglesGrid = repmat(scanAngles, [1, M]);

for t = 2:N % You will start estimating myPose from t=2 using ranges(:,2).

    % 1) Propagate the particles    
    P = P + [std(1).*randn(1,M); std(2).*randn(1,M); std(3).*randn(1,M)];
    corr = zeros(1, M);

    thetas = P(3, :);
    lidarAngles = thetas + scanAnglesGrid;
    d = ranges(:, t);

    startPoint_x = P(1, :);
    startPoint_y = P(2, :);

    endPoint_x = (d.*cos(lidarAngles)) + startPoint_x;
    endPoint_y = (-d.*sin(lidarAngles)) + startPoint_y;

    startIndex_x = ceil(startPoint_x .* myResolution) + myOrigin(1);
    startIndex_y = ceil(startPoint_y .* myResolution) + myOrigin(2);
    endIndex_x = ceil(endPoint_x .* myResolution) + myOrigin(1);
    endIndex_y = ceil(endPoint_y .* myResolution) + myOrigin(2);

    for m=1:M
        startIndex = [startIndex_x(m), startIndex_y(m)];    % 1-by-2
        endIndex = [min(max(endIndex_x(:, m), 1), numCols), min(max(endIndex_y(:, m), 1), numRows)];    % L-by-2

        %startIndex = [min(max(startIndex_x(m), 1), numCols), min(max(startIndex_y(m), 1), numRows)];    % 1-by-2
        %endIndex = [min(max(endIndex_x(:, m), 1), numCols), min(max(endIndex_y(:, m), 1), numRows)];    % L-by-2

        %startIndex = [startIndex_x(m), startIndex_y(m)];    % 1-by-2
        %validLocs = (endIndex_x(:, m) >=1) & (endIndex_x(:, m) < numCols) & (endIndex_y(:, m) >= 1) & (endIndex_y(:, m) < numRows);
        %endIndex = [endIndex_x(validLocs, m), endIndex_y(validLocs, m)];    % L-by-2

        corr(m) = computeCorrelation(map, startIndex, endIndex);
    end

    %W = W .* corr;
    W = W .* (corr - min(corr));
    W = W / sum(W);
    
    [~, maxIndex] = max(W);
    myPose(:, t) = P(:, maxIndex);

    %P = repmat(myPose(:, t), [1, M]);
    %W = ones(1, M) / M;

    neff = (sum(W)^2) / sum(W.^2);
    if neff < (0.8 * M)
        %%{
        P = repmat(myPose(:,t), [1, M]);
        W = ones(1, M) / M;
        %%}

        %{
        % X(find(rand<cumsum(P),1,'first'))
        W_cumsum = cumsum((W - min(W)) / sum(W - min(W)));
        W_copy = W(:, :);
        P_copy = P(:, :);
        for m = 1:M
            sample_index = find(rand() < W_cumsum, 1);
            W(:, m) = W_copy(:, sample_index);
            P(:, m) = P_copy(:, sample_index);
        end
        %}

        %{
        [W_sorted, sorted_index] = sort(W);
        W_cumsum = cumsum(W_sorted);
        P_sorted = P(:, sorted_index);

        for m=1:M
            sample_id = find(W_cumsum >= rand(), 1);
            P(:, m) = P_sorted(:, sample_id);
            W(:, m) = W_sorted(:, sample_id);
        end
        %}
    end

    % 2) Measurement Update 
    %   2-1) Find grid cells hit by the rays (in the grid map coordinate frame)

    %   2-2) For each particle, calculate the correlation scores of the particles

    %   2-3) Update the particle weights
 
    %   2-4) Choose the best particle to update the pose
    
    % 3) Resample if the effective number of particles is smaller than a threshold

    % 4) Visualize the pose on the map as needed
   

end

function corr = computeCorrelation(map, startIndex, endIndex)
    L = size(endIndex, 1);
    if map(startIndex(2), startIndex(1)) >= 0.05
        corr = L * -10;
        return;
    end
    Locc_indices = sub2ind(size(map), endIndex(:, 2), endIndex(:, 1));

    Locc_Mocc_score = 10 * sum(map(Locc_indices) >= 0.55);
    Locc_Mfree_score = -5 * sum(map(Locc_indices) < 0.55);
    corr = Locc_Mocc_score + Locc_Mfree_score;
    %return;

    for l = 1:23:L
        [freex, freey] = bresenham(startIndex(1), startIndex(2), endIndex(l, 1), endIndex(l, 2));

        Lfree_indices = sub2ind(size(map), freey, freex);
        Lfree_Mocc_score = -5 * sum(map(Lfree_indices) >= 0.55);
        Lfree_Mfree_score = 1 * sum(map(Lfree_indices) < 0.55);
        corr = corr + Lfree_Mocc_score + Lfree_Mfree_score;

        %{
        if size(freex, 2) == 0
            corr = corr - 5;
        else
            Lfree_indices = sub2ind(size(map), freey, freex);
            Lfree_Mocc_score = -5 * sum(map(Lfree_indices) >= 0.55);
            Lfree_Mfree_score = 1 * sum(map(Lfree_indices) < 0.55);
            corr = corr + Lfree_Mocc_score + Lfree_Mfree_score;
        end
        %}
    end
end

end

