
im = imread('./parasite.jpg');
gray_im = im2gray(im);
s = size(gray_im);

%--------------------------------------------------------------------------
% tensor voting
%--------------------------------------------------------------------------
tic;
tensors = pixel2tensor(gray_im, 3, 1);
toc;
result = get_eig(tensors);
img = zeros(length(result), 1);
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



%--------------------------------------------------------------------------
% oriented filter
%--------------------------------------------------------------------------
% result = steerable_filter(gray_im,3,0,0,90);
% Im = uint8(255) * ones(s(1),s(2),'uint8');
% Im(:,:) = result;
% imshow(Im);



%--------------------------------------------------------------------------
% oriented filter + tensor voting
%--------------------------------------------------------------------------
% result = steerable_filter(gray_im,3,18,1,1);
% result = reshape(result, [], 1);
% img = zeros(length(result), 1);
% 
% diff = abs(max(result)) - abs(min(result));
% thresh = diff * 0.55 + abs(min(result));
% 
% for i = 1:length(result)
%     if abs(result(i)) > thresh
%         img(i) = 255;
%     else
%         img(i) = 0;
%     end
% end
% 
% img = reshape(img, s(1), s(2));
% 
% Im = uint8(255) * ones(s(1),s(2),'uint8');
% Im(:,:) = img;
% imshow(Im);