function signedImg = add_signature(mainImg, sigPath)
% ADD_SIGNATURE Overlays a signature on the bottom-right corner of an image
% Inputs: 
%   mainImg: The image to be signed (RGB or Grayscale)
%   sigPath: File path to the signature image
% Output:
%   signedImg: The combined image

    % 1. Load Signature
    if ~isfile(sigPath)
        warning('Signature file not found. Returning original image.');
        signedImg = mainImg;
        return;
    end
    sig = imread(sigPath);
    
    % 2. Ensuring Main Image is RGB for colored signature
    if size(mainImg, 3) == 1
        mainImg = cat(3, mainImg, mainImg, mainImg);
    end
    
    % 3. Processing Signature (Make background transparent)
    % Convert signature to grayscale
    if size(sig, 3) == 3
        sigGray = rgb2gray(sig);
    else
        sigGray = sig;
    end
    
    % Create a mask: Black ink (dark pixels) = 1, White paper = 0
    % We assume the signature is dark ink on light background
    mask = sigGray < 200; 
    
    % 4. Resize Signature to fit the main image (25% of width)
    targetWidth = round(size(mainImg, 2) * 0.40); 
    scale = targetWidth / size(sig, 2);
    sigResized = imresize(sig, scale);
    maskResized = imresize(mask, scale);
    
    % 5. Determining Position (Bottom-Right with padding)
    [h, w, ~] = size(mainImg);
    [sH, sW, ~] = size(sigResized);
    
    padding = 10;
    rowStart = h - sH - padding;
    colStart = w - sW - padding;
    
    % Safety check: if signature is bigger than image, skip
    if rowStart < 1 || colStart < 1
        warning('Signature is too big for this image.');
        signedImg = mainImg;
        return;
    end
    
    rowRange = rowStart : (rowStart + sH - 1);
    colRange = colStart : (colStart + sW - 1);
    
    % 6. Super-impose
    signedImg = mainImg;
    
    % Loop through RGB channels
    for c = 1:3
        channel = signedImg(:, :, c);
        sigChannel = sigResized(:, :, min(c, size(sigResized,3)));
        
        % Replacing pixels ONLY where the mask is true (the ink)
        channel(rowRange, colRange) = ...
            channel(rowRange, colRange) .* uint8(~maskResized) + ...
            sigChannel .* uint8(maskResized);
            
        signedImg(:, :, c) = channel;
    end
end