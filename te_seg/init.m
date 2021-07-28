
im = imread('../parasite.jpg');
gray_im = im2gray(im);

s = size(gray_im);

%--------------------------------------------------------------------------
% tensor voting
%--------------------------------------------------------------------------
tic;
tensors = tv_seg(gray_im, 3);
toc;
result = get_eigen(tensors,s(1), s(2));
% tensors = tv_seg(eig_val, 3);
% eig_val = get_eigen(tensors, s(1), s(2));
% for i = 1:9
%     tensors = tv_seg(eig_val, 3);
%     eig_val = get_eigen(tensors, s(1), s(2));
% end

%result = reshape(eig_val, [], 1);
img = zeros(length(result), 1);
% maximum = abs(min(result));
% 
% for i = 1:length(result)
%     img(i) = floor(abs(result(i)) / maximum * 255);
% end

diff = abs(min(result)) - abs(max(result));
thresh = diff * 0.35 + abs(max(result));

for i = 1:length(result)
    if abs(result(i)) > thresh
        img(i) = 255;
    else
        img(i) = 0;
    end
end

img = reshape(img, s(2), s(1)).';

Im = uint8(255) * ones(s(1),s(2),'uint8');
Im(:,:) = img;
imshow(Im);


