function [L, mse, rmse, snr, psnr, cp, nmse, mssim, iter, cor] = ...
    MLEM_FuzzyAD(G, N_true, image, kappa, iter_max, delta_t, control, input)
% MLEM_FUZZYAD  MLEM reconstruction with fuzzy anisotropic diffusion.
%
%   Each iteration: MLEM multiplicative update, then a fuzzy anisotropic
%   diffusion step (anisodiff2D_fuzzy) applied to the current estimate.
%   Same structure as the MLEM_o3 AD variant, with fuzzy AD as the prior.
%
%   control = 0 -> pure MLEM ; control = 1 -> MLEM + FuzzyAD
%   Same I/O signature as MLEM_o3.

    if nargin < 5 || isempty(iter_max), iter_max = 100; end

    image        = double(image);
    num_pixels   = numel(image);
    phantom_size = round(sqrt(num_pixels));
    if phantom_size^2 ~= num_pixels
        error('MLEM_FuzzyAD: image is not a perfect square.');
    end
    image_gt = reshape(image, [phantom_size, phantom_size]);

    G  = G';
    G1 = sum(G, 2);
    L  = max(rand(size(G,1),1), eps);

    niter = 3;     % fuzzy-AD sub-iterations per MLEM step (matches MLEM_o3)

    mse=zeros(iter_max+1,1); rmse=mse; snr=mse; psnr=mse;
    cp=mse; nmse=mse; mssim=mse; cor=mse;

    curr = reshape(L, [phantom_size phantom_size]);
    [mse(1),rmse(1),snr(1),psnr(1),cp(1),nmse(1),mssim(1),cor(1)] = ...
        error_analysis(image_gt, curr);

    for iter = 1:iter_max
        % --- MLEM multiplicative update ---
        N_cal = max(G' * L, eps);
        ratio = N_true ./ N_cal;
        L = (L .* (G * ratio)) ./ max(G1, eps);
        L = max(L, 0);

        if control == 1
            % --- Fuzzy anisotropic diffusion step ---
            Limg = reshape(L, [phantom_size phantom_size]);
            Limg = anisodiff2D_fuzzy(Limg, niter, delta_t, kappa, 2);
            L    = reshape(max(Limg, 0), [phantom_size^2, 1]);
        end

        Limg = reshape(L, [phantom_size phantom_size]);
        [mse(iter+1),rmse(iter+1),snr(iter+1),psnr(iter+1), ...
         cp(iter+1),nmse(iter+1),mssim(iter+1),cor(iter+1)] = ...
            error_analysis(image_gt, Limg);
    end
end
