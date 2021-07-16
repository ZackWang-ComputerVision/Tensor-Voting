% this function can generate two different results
%   1) steerable filter
%   2) steerable filter with tensor voting

% the 5 inputs are:
%   1) M x N image
%   2) define kernel size (3=3x3, 5=5x5, ... , 2n+1 = 2n+1 x 2n+1)
%   3) sigma for decay function (will not affect when using steerable
%   filter)
%   4,5) degree for interpolation function (if they are same number, it
%   refers to use a steerable filter with tensor voting)

% the output is a M x N matrix

function [result] = steerable_filter(pixels, n_neighbours, sigma, deg1, deg2)
    % n_filter = the size of the filter (3x3, 5x5, NxN)
    % d = how many neighbours in each direction
    d = (n_neighbours - 1) / 2;
    padded_img = padarray(pixels,[d d],0,'both');
    dim = size(pixels);
    t = zeros(dim(1) * dim(2), n_neighbours^2);
    if deg1 == deg2
        kernel = steer_tv_kernel(n_neighbours, sigma);
        count = 0;
        for i = d + 1:dim(1) + d
            for j = d + 1:dim(2) + d
                arr = padded_img(i - d: i + d, j - d: j + d);
                flat = double(reshape(arr, [], 1));
                count = count + 1;
                for k = 1:length(flat)
                     p2t = [flat(k), flat(k)];
                     proj_tensor =p2t.*kernel(:, :, k);
                     val = sqrt(proj_tensor(1)^2 + proj_tensor(2)^2);
                     t(count, k) = val;
                end
            end
        end
        t_sum = sum(t,2);
        s = size(pixels);
        result = reshape(t_sum, s(1),s(2)).';
    else
        temp = zeros(n_neighbours^2,1);
        index = 0;
        for y = -d:d
            for x = -d:d
                index = index + 1;
                Gx = -x * exp(-(x^2 + y^2)/2);
                Gy = y * exp(-(x^2 + y^2)/2);
                interpolation = cosd(deg1) * Gx + sind(deg2) * Gy;
                temp(index) = double(interpolation);
            end
        end
        conv = reshape(temp, n_neighbours, n_neighbours).';
        result = imfilter(pixels, conv);
    end
end

