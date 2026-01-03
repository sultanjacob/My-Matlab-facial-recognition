function analyze_lbp(queryImg, matchImg)
% ANALYZE_LBP Performs Texture Analysis using Local Binary Patterns
% Visualizes the texture features and compares histograms.

    % 1. Preprocess (Gray & Resize)
    if size(queryImg, 3) == 3
        qGray = rgb2gray(queryImg);
    else
        qGray = queryImg;
    end
    qGray = imresize(qGray, [256 256]);
    
    if size(matchImg, 3) == 3
        mGray = rgb2gray(matchImg);
    else
        mGray = matchImg;
    end
    mGray = imresize(mGray, [256 256]);

    % 2. Extract LBP Features (FIXED: Only request one output)
    % 'CellSize' splits the image into blocks for better local detail
    % 'Upright', false makes it rotation invariant
    qFeatures = extractLBPFeatures(qGray, 'CellSize', [32 32], 'Upright', false);
    mFeatures = extractLBPFeatures(mGray, 'CellSize', [32 32], 'Upright', false);

    % 3. Calculate Texture Similarity (Chi-Square Distance)
    % Lower value means textures are more similar
    diff = (qFeatures - mFeatures).^2 ./ (qFeatures + mFeatures + eps);
    textureDistance = 0.5 * sum(diff);
    
    % 4. Visualization
    figure('Name', 'LBP Texture Analysis', 'Color', 'w');
    
    % --- Top Row: Input Images Used for LBP ---
    subplot(2, 2, 1);
    imshow(qGray); 
    title('Query Image (Gray)', 'FontWeight', 'bold');
    
    subplot(2, 2, 2);
    imshow(mGray);
    title('Match Image (Gray)', 'FontWeight', 'bold');
    
    % --- Bottom Row: Feature Histograms ---
    subplot(2, 1, 2);
    plot(qFeatures, 'r', 'LineWidth', 1.5); hold on;
    plot(mFeatures, 'b', 'LineWidth', 1.5);
    legend('Query Texture', 'Match Texture');
    title(['Texture Feature Comparison (Distance: ' num2str(textureDistance, '%.4f') ')']);
    grid on;
    xlabel('LBP Feature Vector Index');
    ylabel('Feature Magnitude');
    
    fprintf('LBP Texture Distance: %.4f\n', textureDistance);
end