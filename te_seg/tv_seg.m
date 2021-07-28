% the 3 inputs are:
%   1) M x N image, 
%   2) define kernel size (3=3x3, 5=5x5, ... , 2n+1 = 2n+1 x 2n+1)

% the output is a (M x N x 2 x 2) matrix. Each pixel converts to a 2x2
% matrix and collects information from its neighbour.

function [tensors] = tv_seg(pixels, n_neighbours)
    % d = how many neighbour is on each direction
    d = (n_neighbours - 1) / 2;
    % pad around the image for the kernel to provide a uniform operation
    padded_img = padarray(pixels,[d d],0,'both');
    dim = size(pixels);
    % create an empty multi-dimentional array for the result
    %t = zeros(dim(1) * dim(2), n_neighbours^2, 2, 2);
    t = zeros(2, 2, dim(1) * dim(2), n_neighbours^2);
    
    % call the kernel function
    %kernel = my_kernel(n_neighbours);
    kernel = kernel2(n_neighbours);
    % for getting the pixel's position in an image
    count = 0;
    % go through each pixel in the padded image. Because of the padding, the
    % actual pixel starts at d position and ends at size(image) + d
    for i = d + 1:dim(1) + d
        for j = d + 1:dim(2) + d
            % get neighbours for the pixel
            arr = padded_img(i - d: i + d, j - d: j + d);
            % stretch out to a vector
            flat = double(reshape(arr.', [], 1));
            count = count + 1;
            % apply the kernel to the vector
            for k = 1:length(flat)
                 % convert from a scalar to a 2x2 matrix
                 p2t = [flat(k), 0; 0, flat(k)];
                 % apply the kernel
                 proj_tensor = mtimes(p2t,kernel(:, :, k));
                 %proj_tensor = p2t * kernel(:, :, k);
                 % use the empty matrix to hold the result
                 t(:, :, count, k) = proj_tensor;
            end
        end
    end
    % sum up (collect) the neighbours' information to become a new tensor

    t_sum = sum(t, 4);

    tensors = t_sum;
end



