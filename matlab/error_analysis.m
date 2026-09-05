function [mse, rmse, snr, psnr, CP, nmse, mssim, cor] = error_analysis(I1, I2)
% ERROR_ANALYSIS  Quantitative comparison between reference I1 and test I2.
%
%   Notes:
%   (a) The SSIM dynamic range L is derived from the reference image range
%       (appropriate for images in [0,1]).
%   (b) NMSE is the standard normalised mean-squared error.
%   (c) PSNR/SNR are delegated to PSNR.m / SNR.m.

    I1 = double(I1);
    I2 = double(I2);

    [x, y, ~] = size(I1);

    % ---- MSE / RMSE ----
    mse  = sum(sum((I2 - I1).^2)) / (x * y);
    rmse = sqrt(mse);

    % ---- SNR / PSNR (use corrected helpers) ----
    snr  = SNR(I1, I2);
    psnr = PSNR(I1, I2);

    % ---- Correlation parameter (edge preservation) ----
    h   = fspecial('laplacian');
    Ih1 = imfilter(I1, h, 'replicate');
    Ih2 = imfilter(I2, h, 'replicate');
    mI1 = mean2(Ih1);          % NOTE: mean of the high-pass image, not of I1
    mI2 = mean2(Ih2);
    X1  = sum(sum((Ih1 - mI1) .* (Ih2 - mI2)));
    X2  = sum(sum((Ih1 - mI1).^2));
    X3  = sum(sum((Ih2 - mI2).^2));
    CP  = X1 ./ sqrt(X2 .* X3);

    % ---- Normalised MSE (corrected) ----
    nmse = sum((I2(:) - I1(:)).^2) / sum(I1(:).^2);

    % ---- SSIM with dynamic range matched to the data ----
    L = max(I1(:)) - min(I1(:));
    if L <= 0
        L = 1;
    end
    K      = [0.01 0.03];
    window = fspecial('gaussian', 11, 1.5);
    mssim  = ssim(I1, I2, K, window, L);

    % ---- Pearson correlation ----
    cor = corr2(I1, I2);
end
