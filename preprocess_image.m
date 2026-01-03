function img_out = preprocess_image(img_in)
% PREPROCESS_IMAGE
% Converts image to grayscale, resizes, and smooths it
% Input:  RGB or grayscale image
% Output: Preprocessed grayscale image

    % Converting to grayscale if RGB
    if size(img_in, 3) == 3
        img_gray = rgb2gray(img_in);
    else
        img_gray = img_in;
    end

    % Resizing image for consistency
    img_resized = imresize(img_gray, [256 256]);

    % Applying Gaussian smoothing to reduce noise
    img_out = imgaussfilt(img_resized, 1);

end
