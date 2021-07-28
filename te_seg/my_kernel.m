% the 2 inputs are:
%   1) define kernel size (3=3x3, 5=5x5, ... , 2n+1 = 2n+1 x 2n+1)

% the output is a vector contains n^2 number of 2x2 matrix

function [kernel] = my_kernel(n_neighbours)
    % n_neighbours = size of the kernel (3x3, 5x5, nxn)
    d = (n_neighbours - 1) / 2;
    % precalculate how many neighbours will be included in the kernel
    temp = zeros(2, 2, n_neighbours^2);
    i = 0;
    rad = 0;
    
    for y = -d : d
        for x = -d : d
            i = i + 1;
            if x ~= 0
                rad = atan(-y / x);
            else
                if y < 0
                    rad = 1.5708;
                elseif y > 0
                    rad = 4.7124;
                end
            end
            %degree = rad * 180 / 3.14159;
            projection = [sin(2 * rad)^2, (-sin(2 * rad) * cos(2 * rad));
                          (-sin(2 * rad) * cos(2 * rad)), cos(2 * rad)^2];
%             projection = [sind(degree)^2, sind(degree) * cosd(degree);
%                           sind(2 * degree) * cosd(2 * degree), cosd(degree)^2];

            if x == 0 && y == 0
                projection = [1, 1; 1, 1];
            end
            temp(:, :, i) = projection;   
        end
    end
    kernel = temp;
end

