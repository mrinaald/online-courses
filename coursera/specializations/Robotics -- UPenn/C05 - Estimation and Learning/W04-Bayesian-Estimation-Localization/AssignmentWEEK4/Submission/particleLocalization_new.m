% Robotics: Estimation and Learning 
% WEEK 4
% 
% Complete this function following the instruction. 
function myPose = particleLocalization_new(ranges, scanAngles, map, param)

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
W = ones(1, M) / M;
corr = zeros(1, M);

posDiff = zeros(3, N - 1);

noise = [0.05; 0.05; 5 * (pi / 180)];
debug_plot = 0;

for t = 2:N % You will start estimating myPose from t=2 using ranges(:,2).

    d = ranges(:, t);
    % 1) Propagate the particles
    for m = 1:M
        P(:, m) = P(:,m) + [noise(1) * randn(1); noise(2) * randn(1); noise(3) * randn(1)];

        startPos = [P(1, m), P(2, m)];
        theta = P(3, m);
        lidarAngles = theta + scanAngles;
        endPos = [d.*cos(lidarAngles), -d.*sin(lidarAngles)] + startPos;

        startIndex = ceil(myResolution .* startPos) + myOrigin';
        endIndex = ceil(myResolution .* endPos) + myOrigin';

        corr(m) = findCorrelation(startIndex, endIndex);
    end

    W = W .* (corr - min(corr));
    %W = W .* corr;
    W = W / sum(W);

    [~, maxIndex] = max(W);
    myPose(:, t) = P(:, maxIndex);

    neff = (sum(W)^2) / sum(W.^2);

    if neff < (0.8 * M)
        %{
        P = repmat(myPose(:,t), [1, M]);
        W = ones(1, M) / M;
        %}

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

        %%{
        [W_sorted, sorted_index] = sort(W);
        W_cumsum = cumsum(W_sorted);
        P_sorted = P(:, sorted_index);

        for m=1:M
            sample_id = find(W_cumsum >= rand(), 1);
            P(:, m) = P_sorted(:, sample_id);
            W(:, m) = W_sorted(:, sample_id);
        end
        %%}
    end

    %{
    posDiff(:, t-1) = myPose(:, t) - myPose(:, t-1);
    if mod(t, 50) == 0
        i = t;
        dx = [mean(posDiff(1, :)), std(posDiff(1, :))];
        dy = [mean(posDiff(2, :)), std(posDiff(2, :))];
        dz = [mean(posDiff(3, :)), std(posDiff(3, :))];
        i = [i dx dy dz]
    end
    %}

    % 2) Measurement Update
    %   2-1) Find grid cells hit by the rays (in the grid map coordinate frame)

    %   2-2) For each particle, calculate the correlation scores of the particles

    %   2-3) Update the particle weights
 
    %   2-4) Choose the best particle to update the pose
    
    % 3) Resample if the effective number of particles is smaller than a threshold

    % 4) Visualize the pose on the map as needed
    if debug_plot > 0
        figure(99)
        if exist('h')
            delete(h) % remove old particles
        else
            imagesc(map), hold on,    %xlim([684 686])    %ylim([560 573])
        end
        h = plot(myOrigin(1)+P(1,:)*myResolution,myOrigin(2)+P(2,:)*myResolution, '.r');
        plot(myOrigin(1)+myPose(1,t)*myResolution,myOrigin(2)+myPose(2,t)*myResolution,'.b');
        pause(0.1);
    end

end

    function c = findCorrelation(startIndex, endIndex)
        %%{
        if (startIndex(1) < 1 || startIndex(1) > numCols || startIndex(2) < 1 || startIndex(2) > numCols)
            % Start Index is out of map
            c = 0;
            %return;
        end
        %%}

        endIndex = [min(max(endIndex(:, 1), 1), numCols), min(max(endIndex(:, 2), 1), numRows)];

        Locc_indices = sub2ind(size(map), endIndex(:, 2), endIndex(:, 1));
        
        %{
        occ_score = 10 * sum(map(indices) > 0.55);
        free_score = -10 * sum(map(indices) < 0.05);
        out_of_map_score = 0 * sum(map(indices) >= 0.05 & map(indices) <= 0.55);
        c = occ_score + free_score + out_of_map_score;
        %}

        Locc_Mocc_score = 10 * sum(map(Locc_indices) >= 0.55);
        Locc_Mfree_score = -5 * sum(map(Locc_indices) < 0.05);
        c = Locc_Mocc_score + Locc_Mfree_score;

        %{
        L = size(endIndex, 1);
        for l = 1:L
            [freex, freey] = bresenham(startIndex(1), startIndex(2), endIndex(l, 1), endIndex(l, 2));
            Lfree_indices = sub2ind(size(map), freey, freex);
            Lfree_Mocc_score = -5 * sum(map(Lfree_indices) > 0.55);
            Lfree_Mfree_score = 1 * sum(map(Lfree_indices) < 0.05);
            c = c + Lfree_Mocc_score + Lfree_Mfree_score;
        end
        %}
    end

end