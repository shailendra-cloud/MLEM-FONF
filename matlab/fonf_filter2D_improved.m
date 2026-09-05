function img_out = fonf_filter2D_improved(img_in, params)
% FONF_FILTER2D_IMPROVED
%   Fractional-Order Multi-Notch Filter in 2D frequency domain.
%   Uses mild fractional smoothing + optional radial notches.
%
% INPUTS:
%   img_in  : 2D image (double)
%   params  : struct with fields:
%       .alpha        : fractional order (e.g., 1.0–1.3)
%       .lambda_frac  : smoothing strength
%       .notch_freqs  : vector of normalized radial freqs (0–0.5) [optional]
%       .notch_widths : same size as notch_freqs [optional]
%       .beta_spec    : global spectral mixing factor (0–1)
%
% OUTPUT:
%   img_out : filtered image (same size as img_in)

    % Default output = input (so it is ALWAYS assigned)
    img_out = img_in;

    % Basic checks
    if isempty(img_in)
        return;
    end

    img_in = double(img_in);
    [M, N] = size(img_in);
    if M == 0 || N == 0
        return;
    end

    % defaults for params fields
    if ~isfield(params, 'alpha'),       params.alpha = 1.1;      end
    if ~isfield(params, 'lambda_frac'), params.lambda_frac = 5e-4; end
    if ~isfield(params, 'notch_freqs'), params.notch_freqs = []; end
    if ~isfield(params, 'notch_widths'),params.notch_widths = [];end
    if ~isfield(params, 'beta_spec'),   params.beta_spec = 0.6;  end

    alpha       = params.alpha;
    lambda_frac = params.lambda_frac;
    notch_freqs = params.notch_freqs;
    notch_widths= params.notch_widths;
    beta_spec   = params.beta_spec;

    try
        % 2D FFT and frequency grid
        F = fftshift(fft2(img_in));

        [u, v] = meshgrid( (-floor(N/2):ceil(N/2)-1)/N, ...
                           (-floor(M/2):ceil(M/2)-1)/M );
        rho = sqrt(u.^2 + v.^2);        % radial frequency

        % Fractional-order low-pass
        H_frac = 1 ./ (1 + lambda_frac * (rho.^alpha));

        % Optional multi-notch
        H_notch = ones(size(rho));
        if ~isempty(notch_freqs)
            if isempty(notch_widths)
                notch_widths = 0.02 * ones(size(notch_freqs));
            end
            for k = 1:numel(notch_freqs)
                f0    = notch_freqs(k);
                sigma = notch_widths(k);
                notch_k = 1 - exp( - ( (rho - f0).^2 ) / (2*sigma^2) );
                H_notch = H_notch .* notch_k;
            end
        end

        % Combined FONF transfer function
        H = H_frac .* H_notch;

        % Spectral mixing
        H = (1 - beta_spec) + beta_spec * H;

        % Apply filter
        F_filt  = H .* F;
        img_out = real(ifft2(ifftshift(F_filt)));

        % Cleanup
        img_out(~isfinite(img_out)) = 0;

    catch ME
        warning('fonf_filter2D_improved: error "%s". Returning input image.', ME.message);
        img_out = img_in;
    end

end
