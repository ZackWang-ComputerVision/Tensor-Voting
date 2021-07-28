% it takes in the result from pixel2tensor directly and generate
% eigenvalues for each matrix (pixel tensor)

function [eigvalues] = get_eigen(arr,s1,s2)
    temp = zeros(length(arr), 1);
    for i = 1:length(arr)
        
        m = reshape(arr(:, :, i, 1),[2,2]);
        eig_v = eig(m);
        temp(i) = eig_v(1) - eig_v(2);
    end
    %eigvalues = reshape(temp, s1, s2);
    eigvalues = temp;
end

