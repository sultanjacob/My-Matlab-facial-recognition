clc;
clear;
close all;


% 1. SETTING UP PATHS

dbPath = 'FaceDB/';
queryFolder = 'query/';

% Auto-load Query Image
qDir = dir(fullfile(queryFolder, '*.jpg'));
if isempty(qDir), error('No query image found!'); end
queryPath = fullfile(queryFolder, qDir(1).name);

% Loading Database Images
dbDir = dir(fullfile(dbPath, '*.jpg'));
if length(dbDir) < 2
    error('Need at least 2 images in FaceDB to perform this comparison!');
end


% 2. SELECTING IMAGES FOR COMPARISON

% Reference = The Query Image
refImg = imread(queryPath);
refImg = preprocess_for_metric(refImg); % Resize & Gray

% We need to find the Best Match and Worst Match to show contrast
scores = [];
for i = 1:length(dbDir)
    tempImg = imread(fullfile(dbPath, dbDir(i).name));
    tempImg = preprocess_for_metric(tempImg);
   
    % we use basic MSE to find best/worst quickly
    scores(i) = immse(tempImg, refImg); 
end

[~, bestIdx] = min(scores); % Lowest Error = Best Match
[~, worstIdx] = max(scores); % Highest Error = Worst Match

imgMatch = imread(fullfile(dbPath, dbDir(bestIdx).name));
imgMismatch = imread(fullfile(dbPath, dbDir(worstIdx).name));

% Preprocess them to be identical size
imgMatch = preprocess_for_metric(imgMatch);
imgMismatch = preprocess_for_metric(imgMismatch);


% 3. CALCULATING METRICS

metrics = {'SSIM', 'MSE', 'Chi', 'H-I', 'PSNR'};
results = [];

% --- Calculate for Mismatch (Left Column) ---
results(1,1) = ssim(imgMismatch, refImg);
results(2,1) = immse(imgMismatch, refImg);
results(3,1) = chi_square(imgMismatch, refImg);
results(4,1) = hist_intersection(imgMismatch, refImg);
results(5,1) = psnr(imgMismatch, refImg);

% --- Calculate for Match (Right Column) ---
results(1,2) = ssim(imgMatch, refImg);
results(2,2) = immse(imgMatch, refImg);
results(3,2) = chi_square(imgMatch, refImg);
results(4,2) = hist_intersection(imgMatch, refImg);
results(5,2) = psnr(imgMatch, refImg);


% 4. VISUALIZING

f = figure('Name', 'Similarity Metrics Analysis', 'Color', 'w', ...
    'Position', [100, 100, 900, 800]);

rows = 5;
cols = 3;

for i = 1:rows
    % --- Left Column: Mismatch Image ---
    subplot(rows, cols, (i-1)*3 + 1);
    imshow(imgMismatch);
    title(sprintf('%s: %.4f', metrics{i}, results(i,1)), 'FontWeight','bold');
    
    % --- Middle Column: Reference Image ---
    subplot(rows, cols, (i-1)*3 + 2);
    imshow(refImg);
    if i == 1, title('REFERENCE', 'Color','r'); end
    
    % --- Right Column: Match Image ---
    subplot(rows, cols, (i-1)*3 + 3);
    imshow(imgMatch);
    title(sprintf('%s: %.4f', metrics{i}, results(i,2)), 'FontWeight','bold');
end


% HELPER FUNCTIONS


function imgOut = preprocess_for_metric(imgIn)
    % Metrics require images to be grayscale and EXACT same size
    if size(imgIn, 3) == 3
        imgIn = rgb2gray(imgIn);
    end
    imgOut = imresize(imgIn, [256 256]); % Standardize size
end

function d = chi_square(img1, img2)
    % Calculate normalized histograms
    h1 = imhist(img1) ./ numel(img1);
    h2 = imhist(img2) ./ numel(img2);
    % Chi-Square formula
    idx = (h1 + h2) > 0; % Avoid divide by zero
    d = sum( (h1(idx) - h2(idx)).^2 ./ (h1(idx) + h2(idx)) ) * 0.5;
end

function d = hist_intersection(img1, img2)
    % Calculate normalized histograms
    h1 = imhist(img1) ./ numel(img1);
    h2 = imhist(img2) ./ numel(img2);
    % Intersection formula
    d = sum(min(h1, h2));
end