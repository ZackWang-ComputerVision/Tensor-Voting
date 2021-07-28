% the 2 inputs are:
%   1) define kernel size (3=3x3, 5=5x5, ... , 2n+1 = 2n+1 x 2n+1)
%   2) sigma for decay function (like weight)

% the output is a vector contains n^2 number of 2x2 matrix

function [kernel] = kernel2(n_neighbours)
    % n_neighbours = size of the kernel (3x3, 5x5, nxn)
    d = (n_neighbours - 1) / 2;
    % precalculate how many neighbours will be included in the kernel
    temp = zeros(2, 2, n_neighbours^2);
    i = 0;
    projection = [1, 1; 1, 1];
    for y = -d : d
        for x = -d : d
            i = i + 1;
            % only collects information within +45 degree and -45 degree
            % and handle the center pixel situation separately
            if abs(y) <= abs(x) && x ~= 0
                rad = atan(-y / x);
                degree = rad * 180 / 3.14159;
                projection = [sind(2 * degree)^2, (sind(2 * degree) * cosd(2 * degree));
                              (sind(2 * degree) * cosd(2 * degree)), cosd(2 * degree)^2];
            elseif abs(y) > abs(x) && x ~= 0
                rad = atan(-y / x);
                degree = rad * 180 / 3.14159;
                projection = [sind(degree)^2, (sind(degree) * cosd(degree));
                              (sind(degree) * cosd(degree)), cosd(degree)^2];
            elseif x == 0
                if y < 0
                    degree = 90;
                    projection = [sind(degree)^2, (sind(degree) * cosd(degree));
                              (sind(degree) * cosd(degree)), cosd(degree)^2];
                elseif y > 0
                    degree = 270;
                    projection = [sind(degree)^2, (sind(degree) * cosd(degree));
                              (sind(degree) * cosd(degree)), cosd(degree)^2];
                else
                    projection = [1,1;1,1];
                end
            end
            temp(:, :, i) = projection;
        end
    end
    kernel = temp;
    
end



