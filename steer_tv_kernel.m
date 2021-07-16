function [kernel] = steer_tv_kernel(n_neighbours, sigma)
    d = (n_neighbours - 1) / 2;

    temp = zeros(1, 2, n_neighbours^2);
    c = -16 * log(0.1) * (sigma - 1) / pi^2;
    i = 0;
    for y = -d : d
        for x = -d : d
            i = i + 1;
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
                DF = exp(-(s^2 + c * k^2)/sigma^2);
                
                % if no decay function, comment the line above and
                % uncomment the line below
                %DF = 1;
                
                % decay function with projection [-sin(2θ), cos(2θ)]
                % and with the integration for ball component
                %projection = [
                %    DF * (rad/2 - sin(4 * rad)/8 + sin(2 * rad)^2), DF * (- sin(2 * rad) * cos(2 * rad) + (cos(2 * rad)^2) / 4);
                %    DF * (- sin(2 * rad) * cos(2 * rad) + (cos(2 * rad)^2) / 4), DF * (rad/2 + sin(4 * rad)/8 + cos(2 * rad)^2)
                %];
                
                projection = [DF * sin(2 * rad)^2, DF * (- (sin(2 * rad) * cos(2 * rad)));
                               DF * (- (sin(2 * rad) * cos(2 * rad))), DF * cos(2 * rad)^2];
                
                Gx = -x * exp(-(x^2 + y^2)/2);
                Gy = y * exp(-(x^2 + y^2)/2);

                proj_with_gaussian = [Gy, Gx]*projection; 

                temp(:, :, i) = proj_with_gaussian;
            % the following elseif is to set the center pixel of the
            % kernel to [1, 1; 1, 1] with no projection
            elseif abs(y) <= abs(x) && x == 0
                temp(:, :, i) = [1, 1];
            end
        end
    end
    kernel = temp;
end

