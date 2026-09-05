function [L, mse, rmse, snr, psnr, cp, nmse, mssim, iter, cor] = ...
    MLEM_TV(G, N_true, image, kappa, iter_max, delta_t, control, input, tv_weight)
% MLEM_TV  MLEM reconstruction with Total-Variation regularization.
%
%   Each iteration: MLEM multiplicative update, then a TV denoising step
%   (Chambolle dual algorithm) applied to the current estimate. This is a
%   standard, genuine TV-regularized EM scheme (same structure as the
%   MLEM_o3 AD variant, with TV in place of anisotropic diffusion).
%
%   control = 0 -> pure MLEM ; control = 1 -> MLEM + TV
%   Same I/O signature as MLEM_o3.

    if nargin < 9 || isempty(tv_weight), tv_weight = 0.05; end
    if nargin < 5 || isempty(iter_max),  iter_max = 100;   end

    image        = double(image);
    num_pixels   = numel(image);
    phantom_size = round(sqrt(num_pixels));
    if phantom_size^2 ~= num_pixels
        error('MLEM_TV: image is not a perfect square.');
    end
    image_gt = reshape(image, [phantom_size, phantom_size]);

    G  = G';
    G1 = sum(G, 2);
    L  = max(rand(size(G,1),1), eps);

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
            % --- TV regularization step ---
            Limg = reshape(L, [phantom_size phantom_size]);
            Limg = tv_denoise_chambolle(Limg, tv_weight, 20);
            L    = reshape(max(Limg, 0), [phantom_size^2, 1]);
        end

        Limg = reshape(L, [phantom_size phantom_size]);
        [mse(iter+1),rmse(iter+1),snr(iter+1),psnr(iter+1), ...
         cp(iter+1),nmse(iter+1),mssim(iter+1),cor(iter+1)] = ...
            error_analysis(image_gt, Limg);
    end
end

% =====================================================================
function u = tv_denoise_chambolle(g, weight, niter)
% Chambolle (2004) dual TV denoiser; unconditionally stable (tau<=1/8).
% Operates on a [0,1]-normalised copy so the weight is scale-invariant.
    gmin = min(g(:));
    span = max(max(g(:)) - gmin, eps);
    gn   = (g - gmin) / span;

    tau = 1/8;
    px  = zeros(size(gn));
    py  = zeros(size(gn));
    for k = 1:niter
        div_p = [px(:,1), diff(px,1,2)] + [py(1,:); diff(py,1,1)];
        s  = div_p - gn / weight;
        sx = [diff(s,1,2), zeros(size(s,1),1)];
        sy = [diff(s,1,1); zeros(1,size(s,2))];
        denom = 1 + tau * sqrt(sx.^2 + sy.^2);
        px = (px + tau * sx) ./ denom;
        py = (py + tau * sy) ./ denom;
    end
    div_p = [px(:,1), diff(px,1,2)] + [py(1,:); diff(py,1,1)];
    u = (gn - weight * div_p) * span + gmin;
end
