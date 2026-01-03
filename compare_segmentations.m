function compare_segmentations(imgIn)
% COMPARE_SEGMENTATIONS
% Visualizes segmentation techniques: Ref, SVM, Cropped, Kmeans, HSV
% Matches Distinction Criteria

    % 1. Prepare Reference Image
    if size(imgIn, 3) == 1
        img = cat(3, imgIn, imgIn, imgIn);
    else
        img = imgIn;
    end
    img = imresize(img, [256 256]);
    imgGray = rgb2gray(img);

    
    % A. SVM-Style(Thresholding)
    
    bw_svm = imbinarize(imgGray, 'adaptive', 'Sensitivity', 0.6);
    bw_svm = imfill(bw_svm, 'holes');
    bw_svm = bwareaopen(bw_svm, 100);
    
    imgSVM = img;
    for c = 1:3
        ch = imgSVM(:,:,c);
        ch(~bw_svm) = 0;
        imgSVM(:,:,c) = ch;
    end

   
    % B. Cropped (Face Detection)
    
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, img);
    
    if ~isempty(bbox)
        [~, idx] = max(bbox(:,3));
        bbox = bbox(idx, :);
        imgCropped = imcrop(img, bbox);
        imgCropped = imresize(imgCropped, [256 256]);
    else
        imgCropped = img; 
    end

    
    % C. K-Means Clustering
    
    numColors = 3;
    L = imsegkmeans(img, numColors);
    centerPixel = L(128, 128); 
    maskKmeans = (L == centerPixel);
    
    imgKmeans = img;
    for c = 1:3
        ch = imgKmeans(:,:,c);
        ch(~maskKmeans) = 0;
        imgKmeans(:,:,c) = ch;
    end

    
    % D. HSV Skin Detection
    
    % Convert to HSV color space
    imgHSV = rgb2hsv(img);
    h = imgHSV(:,:,1); % Hue
    s = imgHSV(:,:,2); % Saturation
    
    % Defining Skin Thresholds. This is because Skin usually has low Hue and med Saturation
    % H < 0.1 or H > 0.9 (Red/Orange tones)
    % S > 0.15 and S < 0.9 (Not too gray, not too vivid)
    maskHSV = (h < 0.11 | h > 0.9) & (s > 0.15 & s < 0.9);
    
    % Clean up noise
    maskHSV = imfill(maskHSV, 'holes');
    maskHSV = bwareaopen(maskHSV, 200);
    
    imgSkin = img;
    for c = 1:3
        ch = imgSkin(:,:,c);
        ch(~maskHSV) = 0;
        imgSkin(:,:,c) = ch;
    end

    
    % VISUALIZATION GRID
    
    figure('Name', 'Segmentation Techniques Comparison', 'Color', 'w');
    
    subplot(1, 5, 1); imshow(img); title('Original');
    subplot(1, 5, 2); imshow(imgSVM); title('Thresholding');
    subplot(1, 5, 3); imshow(imgCropped); title('Face Crop');
    subplot(1, 5, 4); imshow(imgKmeans); title('K-Means');
    subplot(1, 5, 5); imshow(imgSkin); title('HSV (Skin)');
end