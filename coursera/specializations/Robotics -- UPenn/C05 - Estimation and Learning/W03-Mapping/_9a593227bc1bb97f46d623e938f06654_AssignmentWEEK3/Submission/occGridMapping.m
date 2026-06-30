% Robotics: Estimation and Learning 
% WEEK 3
% 
% Complete this function following the instruction. 
function myMap = occGridMapping(ranges, scanAngles, pose, param)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Parameters 
% 
% the number of grids for 1 meter.
myResol = param.resol;
% the initial map size in pixels
myMap = zeros(param.size);
% the origin of the map in pixels
myorigin = param.origin; 

% % 4. Log-odd parameters 
lo_occ = param.lo_occ;
lo_free = param.lo_free; 
lo_max = param.lo_max;
lo_min = param.lo_min;

N = size(pose,2);
for t = 1:N % for each time,
    % Find start and end points of the ray
    startPoint = [pose(1,t), pose(2,t)];
    theta = pose(3, t);
    lidarAngles = theta + scanAngles;
    d = ranges(:, t);

    %endPoint = [d.*cos(lidarAngles) + startPoint(1), -d.*size(lidarAngles) + startPoint(2)];
    endPoint = [d.*cos(lidarAngles), -d.*sin(lidarAngles)] + startPoint;
    
    startIndex = ceil(startPoint .* myResol) + myorigin';
    %endIndex = ceil(endPoint ./ myResol) + myorigin';
    endIndex = ceil(endPoint .* myResol) + myorigin';
      
    % Find grids hit by the rays (in the gird map coordinate
    L = size(endIndex, 1);
    for l = 1:L
        %sp = startPoint
        %ep = [endIndex(l, 1) endIndex(l, 2)]
        [freex, freey] = bresenham(startIndex(1), startIndex(2), endIndex(l, 1), endIndex(l, 2));
        
        %minFreex = min(freex)
        %maxFreex = max(freex)
        %minFreey = min(freey)
        %maxFreey = max(freey)

        % Find occupied-measurement cells and free-measurement cells
        free = sub2ind(size(myMap), freey, freex);
        occ = sub2ind(size(myMap), endIndex(l, 2), endIndex(l, 1));
        
        % Update the log-odds
        myMap(occ) = myMap(occ) + param.lo_occ;
        myMap(free) = myMap(free) - param.lo_free;

        % Saturate the log-odd values
        myMap(occ) = min(myMap(occ), param.lo_max);
        myMap(free) = max(myMap(free), param.lo_min);
    end
    

    % Visualize the map as needed
   

end

end

