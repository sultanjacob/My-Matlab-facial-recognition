clc;
clear;
close all;


% 1. PARAMETERS & PATHS

recognitionThreshold = 0.675; 
numTopMatches = 4; 

dbPath = 'FaceDB/';   
queryFolder = 'query/'; 

% Auto-detect the query image
queryDir = dir(fullfile(queryFolder, '*.jpg'));

queryPath = fullfile(queryFolder, queryDir(1).name);


% 2. LOAD & VISUALIZE DATABASE

dbImages = dir(fullfile(dbPath, '*.jpg'));
numDB = length(dbImages);

fprintf('Number of database images: %d\n', numDB);

figure('Name','Database Overview');
for i = 1:numDB
    img = imread(fullfile(dbPath, dbImages(i).name));
    subplot(2, ceil(numDB/2), i);
    imshow(img);
    title(dbImages(i).name, 'Interpreter','none');
end


% 3. VISUALIZATION CHECK: PREPROCESSING

testImg = imread(fullfile(dbPath, dbImages(1).name));
prepImg = preprocess_image(testImg);

figure('Name','Preprocessing Test');
subplot(1,2,1); imshow(testImg); title('Original Image');
subplot(1,2,2); imshow(prepImg); title('Preprocessed Image');


% 4. VISUALIZATION CHECK: SEGMENTATION

faceMask = segment_face(prepImg);

figure('Name','Face Segmentation');
subplot(1,3,1); imshow(prepImg); title('Preprocessed');
subplot(1,3,2); imshow(faceMask); title('Mask');
subplot(1,3,3); imshow(prepImg .* uint8(faceMask)); title('Extracted Face');


% 5. FEATURE EXTRACTION

fprintf('Extracting features...\n');

% Process Query Image
queryImg = imread(queryPath);
try
    qFace = preprocess_image(queryImg);
    qFeatures = extract_features(qFace);
    fprintf('Query feature length: %d\n', length(qFeatures));
catch
    error('Could not process query image.');
end

% Process Database Images
dbFeatures = [];
dbNames = {};

for i = 1:numDB
    filename = dbImages(i).name;
    img = imread(fullfile(dbPath, filename));
    
    try
        face = preprocess_image(img);
        feat = extract_features(face);
        dbFeatures = [dbFeatures; feat];
        dbNames{i} = filename;
    catch
        dbFeatures = [dbFeatures; zeros(1, length(qFeatures))]; 
        dbNames{i} = [filename ' (Error)'];
    end
end


% 6. CALCULATE RECOGNITION SIMILARITY

distances = zeros(numDB,1);
for i = 1:numDB
    distances(i) = norm(dbFeatures(i,:) - qFeatures);
end

% Sort results
[sortedDistances, sortedIdx] = sort(distances);
bestMatchName = dbNames{sortedIdx(1)};
minDist = sortedDistances(1);
confidence = (1 / (1 + minDist)) * 100;

% Output Recognition Text
fprintf('\n---------------------------------------\n');
if confidence > (recognitionThreshold * 100)
    fprintf('✅ MATCH FOUND: %s (Conf: %.1f%%)\n', bestMatchName, confidence);
else
    fprintf('⚠️ UNKNOWN PERSON (Closest: %s)\n', bestMatchName);
end
fprintf('---------------------------------------\n');


% 7. VISUALIZE RESULT WITH INTERACTIVE SIGNATURE

bestImg = imread(fullfile(dbPath, bestMatchName));
figure('Name','Recognition Result');

% 1. Showing the Query and the Best Match (Unsigned first)
subplot(1,2,1); imshow(queryImg); title('Query');
ax = subplot(1,2,2); imshow(bestImg); 
title(['Best Match: ' bestMatchName], 'Interpreter', 'none');

% 2. INTERACTIVE SIGNING LOGIC
if confidence > (recognitionThreshold * 100)
    fprintf('Authentication Successful. Please sign the image using your mouse.\n');
    title('Authentication Successful! Please Sign Here -->', 'Color', 'g');
    
    % Activating the freehand drawing tool on the current axis
    h = drawfreehand('Color', 'r', 'LineWidth', 3);
    
    % Wait for you to finish drawing double-click to finish and wait for one second)
    fprintf('Draw your signature. Double-click the line when done.\n');
    pause(1); 
    
    % Burning the signature into the image permanently
    mask = createMask(h);
    
    redChannel = bestImg(:,:,1);
    greenChannel = bestImg(:,:,2);
    blueChannel = bestImg(:,:,3);
    
    % Making drawn pixels Red
    redChannel(mask) = 255;
    greenChannel(mask) = 0;
    blueChannel(mask) = 0;
    
    bestImgSigned = cat(3, redChannel, greenChannel, blueChannel);
    
    % Updating the display
    imshow(bestImgSigned);
    title(['Signed & Verified: ' bestMatchName], 'Interpreter', 'none', 'Color', 'b');
    
    fprintf('Signature captured and saved.\n');
else
    title('Access Denied: Signature Not Allowed', 'Color', 'r');
end


% 8. DISTANCE BAR CHART

figure('Name','Distance Scores');
bar(sortedDistances);
xticks(1:numDB);
xticklabels(dbNames(sortedIdx));
xtickangle(45);
ylabel('Euclidean Distance');
title('Similarity (Lower Bar = Better Match)');
grid on;

hold on;
% Calculating the distance equivalent of our threshold
thresholdDistance = (1 / recognitionThreshold) - 1; 
yline(thresholdDistance, 'r--', 'LineWidth', 2, 'Label', 'Cut-off Threshold');
hold off;


% 9. CONFUSION MATRIX

groundTruth = zeros(numDB,1); 
groundTruth(1) = 1;  
groundTruth(2) = 1;

predictions = zeros(numDB,1);
for i = 1:numDB
    score = 1 / (1 + distances(i));
    if score >= recognitionThreshold
        predictions(i) = 1;
    end
end

figure('Name','Confusion Matrix');
confusionchart(confusionmat(groundTruth, predictions), {'other','me'});
title('Confusion Matrix');


% 10. SEGMENTATION COMPARISON

fprintf('Generating Segmentation Visualizations...\n');
compare_segmentations(queryImg);


% 11. ADVANCED SIMILARITY METRICS

fprintf('Calculating Advanced Metrics...\n');

refImgFormatted = preprocess_for_metric(queryImg);
matchImgRaw = imread(fullfile(dbPath, bestMatchName));
matchImgFormatted = preprocess_for_metric(matchImgRaw);

worstIdx = sortedIdx(end); 
worstMatchName = dbNames{worstIdx};
mismatchImgRaw = imread(fullfile(dbPath, worstMatchName));
mismatchImgFormatted = preprocess_for_metric(mismatchImgRaw);

metrics = {'SSIM', 'MSE', 'Chi', 'H-I', 'PSNR'};
results = zeros(5, 2); 

% Mismatch Column
results(1,1) = ssim(mismatchImgFormatted, refImgFormatted);
results(2,1) = immse(mismatchImgFormatted, refImgFormatted);
results(3,1) = calc_chi_square(mismatchImgFormatted, refImgFormatted);
results(4,1) = calc_hist_intersect(mismatchImgFormatted, refImgFormatted);
results(5,1) = psnr(mismatchImgFormatted, refImgFormatted);

% Match Column
results(1,2) = ssim(matchImgFormatted, refImgFormatted);
results(2,2) = immse(matchImgFormatted, refImgFormatted);
results(3,2) = calc_chi_square(matchImgFormatted, refImgFormatted);
results(4,2) = calc_hist_intersect(matchImgFormatted, refImgFormatted);
results(5,2) = psnr(matchImgFormatted, refImgFormatted);

figure('Name', 'Detailed Similarity Metrics', 'Color', 'w');
rows = 5; 
cols = 3;

for i = 1:rows
    % Left: Mismatch
    subplot(rows, cols, (i-1)*3 + 1);
    imshow(mismatchImgFormatted);
    title(sprintf('%s: %.4f', metrics{i}, results(i,1)), 'FontWeight','bold');
    if i==1, ylabel('Worst Match', 'FontWeight','bold'); end
    
    % Center: Reference
    subplot(rows, cols, (i-1)*3 + 2);
    imshow(refImgFormatted);
    if i == 1, title('REFERENCE', 'Color','r', 'FontSize', 12); end
    
    % Right: Match
    subplot(rows, cols, (i-1)*3 + 3);
    imshow(matchImgFormatted);
    title(sprintf('%s: %.4f', metrics{i}, results(i,2)), 'FontWeight','bold');
    if i==1, ylabel('Best Match', 'FontWeight','bold'); end
end


% 12. LOCAL HELPER FUNCTIONS

function imgOut = preprocess_for_metric(imgIn)
    if size(imgIn, 3) == 3
        imgIn = rgb2gray(imgIn);
    end
    imgOut = imresize(imgIn, [256 256]);
end

function d = calc_chi_square(img1, img2)
    h1 = imhist(img1) ./ numel(img1);
    h2 = imhist(img2) ./ numel(img2);
    idx = (h1 + h2) > 0;
    d = sum( (h1(idx) - h2(idx)).^2 ./ (h1(idx) + h2(idx)) ) * 0.5;
end

function d = calc_hist_intersect(img1, img2)
    h1 = imhist(img1) ./ numel(img1);
    h2 = imhist(img2) ./ numel(img2);
    d = sum(min(h1, h2));
end

% 13. PERSONAL LEARNT FEATURE: LBP TEXTURE ANALYSIS

fprintf('Generating LBP Texture Analysis...\n');

% Loading the raw images again 
imgQ = imread(queryPath);
imgM = imread(fullfile(dbPath, bestMatchName));

% Running the analysis
analyze_lbp(imgQ, imgM);