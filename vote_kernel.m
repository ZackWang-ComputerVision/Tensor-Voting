% the 2 inputs are:
%   1) define kernel size (3=3x3, 5=5x5, ... , 2n+1 = 2n+1 x 2n+1)
%   2) sigma for decay function (like weight)

% the output is a vector contains n^2 number of 2x2 matrix

function [kernel] = vote_kernel(n_neighbours, sigma)
    % n_neighbours = size of the kernel (3x3, 5x5, nxn)
    d = (n_neighbours - 1) / 2;
    % precalculate how many neighbours will be included in the kernel
    temp = zeros(2, 2, n_neighbours^2);
    c = -16 * log(0.1) * (sigma - 1) / pi^2;
    i = 0;
    for y = -d : d
        for x = -d : d
            i = i + 1;
            % only collects information within +45 degree and -45 degree
            % and handle the center pixel situation separately
            if abs(y) <= abs(x) && x ~= 0
                % l = length between p and q
                l = sqrt(y^2 + x^2);
                rad = atan(-y / x);
                % curve length
                s = (rad * l / sin(rad));
                if rad == 0
                    s = l;
                end
                % curve curvature
                k = 2 * sin(rad) / l;
                % decay function
%                DF = exp(-(s^2 + c * k^2)/sigma^2);

                % if no decay function, comment the line above and
                % uncomment the line below
                DF = 1;
                
                % decay function with projection [-sin(2θ), cos(2θ)]
                % and with the integration for ball component
                %degree = rad * 180 / 3.14159;
                projection = [
                    DF * (rad/2 - sin(4 * rad)/8 + sin(2 * rad)^2), DF * (- sin(2 * rad) * cos(2 * rad) + (cos(2 * rad)^2) / 4);
                    DF * (- sin(2 * rad) * cos(2 * rad) + (cos(2 * rad)^2) / 4), DF * (rad/2 + sin(4 * rad)/8 + cos(2 * rad)^2)
                ];
                
%                 projection = [DF * sin(2 * rad)^2, DF * (- sin(2 * rad) * cos(2 * rad));
%                               DF * (- sin(2 * rad) * cos(2 * rad)), DF * cos(2 * rad)^2];
                temp(:, :, i) = projection;             
            % the following elseif is to set the center pixel of the
            % kernel to [1, 1; 1, 1] with no projection
            elseif abs(y) <= abs(x) && x == 0
                temp(:, :, i) = [1, 1; 1, 1];
            end
        end
    end
    kernel = temp;
    
end

