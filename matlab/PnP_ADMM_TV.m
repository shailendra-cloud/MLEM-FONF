function [L, mse, rmse, snr, psnr, cp, nmse, mssim, iter_out, cor] = ...
    PnP_ADMM_TV(G, N_true, image, kappa, iter_max, delta_t, control, input, rho, tv_weight)
% PNP_ADMM_TV  Plug-and-Play ADMM reconstruction with a classical TV
% denoiser (training-free baseline).
%
%   The denoiser is Chambolle's dual projection algorithm (unconditionally
%   stable for tau <= 1/8), applied to a [0,1]-normalised image. PnP-ADMM
%   is used in its original, training-free form (Venkatakrishnan et al.,
%   GlobalSIP 2013; cf. Romano et al., SIAM J. Imaging Sci. 2017), which
%   keeps the comparison within the same model-based, training-free class
%   as the proposed FONF method.
%
%   ADMM splitting of   min_f  L(f) + tv_weight * TV(f),  f >= 0 :
%     f-update : MLEM fidelity step, blended toward (v - u)
%     v-update : v = TV_denoise(f + u)        (Chambolle)
%     u-update : u = u + f - v
%
%   Same I/O signature as MLEM_o3 so it slots into the main script.

    if nargin < 10 || isempty(tv_weight), tv_weight = 0.08; end
    if nargin < 9  || isempty(rho),       rho = 1.0;        end
    if nargin < 5  || isempty(iter_max),  iter_max = 100;   end

    image        = double(image);
    num_pixels   = numel(image);
    phantom_size = round(sqrt(num_pixels));
    if phantom_size^2 ~= num_pixels
        error('PnP_ADMM_TV: image is not a perfect square.');
    end
    image_gt = reshape(image, [phantom_size, phantom_size]);

    G  = G';
    G1 = sum(G, 2);

    f = max(rand(size(G,1),1), eps);
    v = f;
    u = zeros(size(f));

    mse=zeros(iter_max+1,1); rmse=mse; snr=mse; psnr=mse;
    cp=mse; nmse=mse; mssim=mse; cor=mse;

    curr = reshape(f, [phantom_size phantom_size]);
    [mse(1),rmse(1),snr(1),psnr(1),cp(1),nmse(1),mssim(1),cor(1)] = ...
        error_analysis(image_gt, curr);

    iter_out = iter_max;
    for iter = 1:iter_max
        % ----- f-update: MLEM fidelity step, pulled toward (v - u) -----
        N_cal = max(G' * f, eps);
        ratio = N_true ./ N_cal;
        fMLEM = (f .* (G * ratio)) ./ max(G1, eps);
        f = (fMLEM + rho * (v - u)) / (1 + rho);
        f = max(f, 0);

        % ----- v-update: stable classical TV denoiser (Chambolle) -----
        fimg = reshape(f + u, [phantom_size phantom_size]);
        vimg = tv_denoise_chambolle(fimg, tv_weight, 30);
        v    = reshape(max(vimg, 0), [phantom_size^2, 1]);

        % ----- dual update -----
        u = u + f - v;

        Limg = reshape(v, [phantom_size phantom_size]);
        [mse(iter+1),rmse(iter+1),snr(iter+1),psnr(iter+1), ...
         cp(iter+1),nmse(iter+1),mssim(iter+1),cor(iter+1)] = ...
            error_analysis(image_gt, Limg);
    end

    L = v;
end

% =====================================================================
function u = tv_denoise_chambolle(g, weight, niter)
% Chambolle (2004) dual algorithm for ROF total-variation denoising.
% Unconditionally stable for tau <= 1/8. Operates on a [0,1]-normalised
% copy of g so the smoothing weight is scale-invariant.
    gmin = min(g(:));
    span = max(max(g(:)) - gmin, eps);
    gn   = (g - gmin) / span;            % normalise to [0,1]

    tau = 1/8;
    px  = zeros(size(gn));
    py  = zeros(size(gn));

    for k = 1:niter
        % divergence of p (Neumann boundaries)
        div_p = [px(:,1), diff(px,1,2)] + [py(1,:); diff(py,1,1)];

        % gradient of (div_p - gn/weight)
        s  = div_p - gn / weight;
        sx = [diff(s,1,2), zeros(size(s,1),1)];
        sy = [diff(s,1,1); zeros(1,size(s,2))];

        denom = 1 + tau * sqrt(sx.^2 + sy.^2);
        px = (px + tau * sx) ./ denom;
        py = (py + tau * sy) ./ denom;
    end

    div_p = [px(:,1), diff(px,1,2)] + [py(1,:); diff(py,1,1)];
    un = gn - weight * div_p;

    u = un * span + gmin;                % undo normalisation
end
