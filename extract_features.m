function features = extract_features(faceImg)
% EXTRACT_FEATURES Extracts feature vector from segmented face
% Input:  faceImg - grayscale face image
% Output: features - 1xN feature vector

% Ensuring grayscale
if size(faceImg,3) == 3
    faceImg = rgb2gray(faceImg);
end

faceImg = im2double(faceImg);

% -------------------------
% Basic Statistical Features
% -------------------------
meanVal = mean(faceImg(:));
stdVal  = std(faceImg(:));

% -------------------------
% Histogram Features
% -------------------------
numBins = 16;
histVals = imhist(faceImg, numBins);
histVals = histVals / sum(histVals); % Normalizing

% -------------------------
% Texture Features (GLCM)
% -------------------------
glcm = graycomatrix(faceImg, ...
    'Offset',[0 1], ...
    'Symmetric',true);

stats = graycoprops(glcm, ...
    {'Contrast','Correlation','Energy','Homogeneity'});

textureFeatures = ...
    [stats.Contrast, stats.Correlation, ...
     stats.Energy, stats.Homogeneity];

% -------------------------
% Final Feature Vector
% -------------------------
features = [meanVal, stdVal, histVals', textureFeatures];

end
