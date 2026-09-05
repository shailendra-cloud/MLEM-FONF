function [L, mse, rmse, snr, psnr, cp, nmse, mssim, iter_out, cor] = ...
    MLEM_FONF(G, N_true, image, kappa, iter_max, delta_t, control, input, fonf_params, lambda)
% MLEM_FONF  Proposed method, implemented to MATCH Algorithm 1 of the paper.
%
%   Per iteration:
%     (1) MLEM multiplicative data-fidelity update
%     (2) FONF spectral filtering  fN = F^{-1}( H_FONF .* F(fMLEM) )
%     (3) Relaxation update         f  = (1 - lambda*dt) fMLEM + (lambda*dt) fN
%     (4) Non-negativity            f  = max(f, 0)
%
%   This function implements the proposed method exactly as Algorithm 1 of
%   the paper and is the routine that produced the reported MLEM+FONF
%   results (see results_stats.csv).
%
%   control = 0 -> pure MLEM (FONF off); control = 1 -> MLEM + FONF.

    if nargin < 10 || isempty(lambda),     lambda = 1.0;  end
    if nargin < 9  || isempty(fonf_params)
        % Defaults: the paper's broadband configuration (K = 0, Table I).
        fonf_params.alpha        = 0.5;          % fractional order (paper, Table I)
        fonf_params.lambda_frac  = 0.5;          % base low-pass strength, as run
        fonf_params.notch_freqs  = [];           % K = 0: no notches for the broadband phantoms
        fonf_params.notch_widths = [];
        fonf_params.beta_spec    = 0.6;
    end
    if nargin < 5 || isempty(iter_max), iter_max = 100; end

    image        = double(image);
    num_pixels   = numel(image);
    phantom_size = round(sqrt(num_pixels));
    if phantom_size^2 ~= num_pixels
        error('MLEM_FONF: image is not a perfect square.');
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

    dt = delta_t;
    relax = lambda * dt;                 % stability needs 0 < lambda*dt < 1
    relax = min(max(relax, 0), 0.999);

    iter_out = iter_max;
    for iter = 1:iter_max
        % (1) MLEM data fidelity
        N_cal  = max(G' * L, eps);
        ratio  = N_true ./ N_cal;
        fMLEM  = (L .* (G * ratio)) ./ max(G1, eps);
        fMLEM  = max(fMLEM, 0);

        if control == 1
            % (2) FONF spectral filtering
            fimg  = reshape(fMLEM, [phantom_size phantom_size]);
            fFONF = fonf_filter2D_improved(fimg, fonf_params);
            fFONF(~isfinite(fFONF)) = 0;

            % (3) relaxation update  (Eq. 5 / Algorithm 1)
            freg  = (1 - relax) * fimg + relax * fFONF;

            % (4) non-negativity
            freg  = max(freg, 0);
            L     = reshape(freg, [phantom_size^2, 1]);
        else
            L = fMLEM;
        end

        Limg = reshape(L, [phantom_size phantom_size]);
        [mse(iter+1),rmse(iter+1),snr(iter+1),psnr(iter+1), ...
         cp(iter+1),nmse(iter+1),mssim(iter+1),cor(iter+1)] = ...
            error_analysis(image_gt, Limg);
    end
end
