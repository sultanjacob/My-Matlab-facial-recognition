function faceMask = segment_face(img)

% Ensuring grayscale
if size(img,3) == 3
    img = rgb2gray(img);
end

% Converting to double
img = im2double(img);

% Adaptive thresholding
bw = imbinarize(img, 'adaptive', ...
    'ForegroundPolarity','bright', ...
    'Sensitivity', 0.5);

% Inverting
bw = imcomplement(bw);

% Morphological cleaning
bw = imfill(bw, 'holes');
bw = bwareaopen(bw, 500);

% Keeping largest connected component
cc = bwconncomp(bw);
numPixels = cellfun(@numel, cc.PixelIdxList);
[~, idx] = max(numPixels);

faceMask = false(size(bw));
faceMask(cc.PixelIdxList{idx}) = true;

end
