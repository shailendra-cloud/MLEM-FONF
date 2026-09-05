function fonf_stripe_demo
% =========================================================================
% FONF multi-notch (K>0) demonstration on a synthetic periodic STRIPE artifact.
%
% Purpose: empirically validate the K>0 regime of the FONF prior described in
% Sec. II-C. A single-frequency stripe produces one conjugate pair of
% localized non-DC peaks in F(f); the data-driven selection rule
% (tau = mu_R + 3*sigma_R on the background-normalized residual spectrum R)
% detects them, places notches (K=2), and the activated H_FONF suppresses the
% artifact while leaving the surrounding spectrum intact.
%
% The notch construction follows Eq. (4) and the data-driven selection rule
% of Sec. II-C; only K differs (K=2 here vs K=0 for the broadband phantoms).
%
% Consistent with paper hyperparameters: alpha = 0.5, sigma_m cap = 0.05.
% =========================================================================

rng(0);

%% --- 1. Clean reference image (use your phantom; placeholder Shepp-Logan) ---
% Replace this with your own clean reconstruction / phantom in [0,1], 256x256.
f_clean = phantom('Modified Shepp-Logan', 256);     % requires Image Proc. Toolbox
f_clean = mat2gray(f_clean);                          % normalize to [0,1]
[Ny, Nx] = size(f_clean);

%% --- 2. Inject a synthetic periodic stripe artifact -----------------------
% Stripe = single 2-D sinusoid. Its DFT is a conjugate pair of deltas at +/-w0.
amp     = 0.12;                 % stripe amplitude (relative to [0,1] range)
theta_s = 25*pi/180;            % stripe orientation (radians)
fnorm   = 0.18;                 % stripe spatial frequency (normalized, in (0,0.5])

[xx, yy] = meshgrid((0:Nx-1)-Nx/2, (0:Ny-1)-Ny/2);
u0 = fnorm*cos(theta_s);  v0 = fnorm*sin(theta_s);     % normalized freq vector
stripe   = amp*cos(2*pi*(u0*xx + v0*yy));
f_corr   = f_clean + stripe;                            % corrupted image

%% --- 3. Forward DFT and background-normalized residual spectrum R ----------
F   = fftshift(fft2(f_corr));
mag = abs(F);

% normalized frequency grid in [-0.5, 0.5]
[fx, fy] = meshgrid(((0:Nx-1)-Nx/2)/Nx, ((0:Ny-1)-Ny/2)/Ny);
rho      = sqrt(fx.^2 + fy.^2);

% radially averaged magnitude spectrum  Sbar(rho)  (the smooth, isotropic background)
nbins  = round(0.7071*max(Nx,Ny));
edges  = linspace(0, max(rho(:)), nbins+1);
Sbar_r = zeros(nbins,1);
binid  = discretize(rho, edges);
for b = 1:nbins
    m = (binid==b);
    if any(m(:)), Sbar_r(b) = mean(mag(m)); end
end
Sbar_r(Sbar_r==0) = eps;
Sbar = Sbar_r(max(binid,1));                            % map back to 2-D
R    = mag ./ reshape(Sbar, size(mag));                 % residual map

% mask out DC / low-frequency core (no notches there)
dc_mask    = rho < 0.04;          % slightly enlarged DC core
R(dc_mask) = 0;
% zero a 3-pixel border (suppress FFT edge artifacts)
R(1:3,:)=0; R(end-2:end,:)=0; R(:,1:3)=0; R(:,end-2:end)=0;

%% --- 4. Data-driven peak detection:  tau = mu_R + 3*sigma_R ----------------
band   = ~dc_mask;
mu_R   = mean(R(band));
sig_R  = std(R(band));
tau    = mu_R + 3*sig_R;

% local maxima above tau (3x3 neighborhood; manual to avoid toolbox dependence)
ismax  = R >= imdilate_local(R) & R > tau;
[py, px] = find(ismax);
peakval  = R(sub2ind(size(R), py, px));
[~, ord] = sort(peakval, 'descend');
py = py(ord);  px = px(ord);

% keep detected peaks (the conjugate pair => K=2 for a single stripe)
Kmax   = 4;                                             % allow up to a few
K      = min(Kmax, numel(py));
fprintf('Detected K = %d notch(es) above tau = %.3f\n', K, tau);

%% --- 5. Build H_FONF (alpha = 0.5) with HWHM-based sigma_m (cap = 0.05) ----
alpha    = 0.5;
sig_cap  = 0.05;                                        % paper's sigma_m cap
H = ones(size(mag));
for m = 1:K
    wx_m = fx(py(m), px(m));   wy_m = fy(py(m), px(m));  % notch centre (norm. freq)
    % HWHM estimate from the local peak, converted to Gaussian sigma, then capped
    hwhm   = estimate_hwhm(R, py(m), px(m), rho);
    sig_m  = min(hwhm/sqrt(2*log(2)), sig_cap);
    dist2  = (fx - wx_m).^2 + (fy - wy_m).^2;
    H = H .* (1 - exp(-dist2./(2*sig_m^2))).^(alpha/2);
end

%% --- 6. Apply the notch and invert ---------------------------------------
F_filt = H .* F;
f_rec  = real(ifft2(ifftshift(F_filt)));
f_rec  = min(max(f_rec,0),1);                           % non-negativity + clip

%% --- 7. Quantify -----------------------------------------------------------
psnr_corr = psnr_local(f_corr, f_clean);
psnr_rec  = psnr_local(f_rec , f_clean);
fprintf('PSNR  corrupted = %.2f dB\n', psnr_corr);
fprintf('PSNR  K>0 notch = %.2f dB   (gain %.2f dB)\n', ...
         psnr_rec, psnr_rec - psnr_corr);

%% --- 8. Figure panel (Fig. 6 of the paper) --------------------------------
figure('Color','w','Position',[100 100 760 520]);
subplot(2,3,1); imagesc(f_clean); axis image off; colormap gray; title('(a) clean');
subplot(2,3,2); imagesc(f_corr ); axis image off; title('(b) + stripe');
subplot(2,3,3); imagesc(f_rec  ); axis image off; title('(c) K>0 notch');
subplot(2,3,4); imagesc(log(1+abs(F))); axis image off; title('(d) |F| corrupted');
hold on; plot(px(1:K), py(1:K), 'r+', 'MarkerSize',10,'LineWidth',1.5);
subplot(2,3,5); imagesc(H); axis image off; title('(e) H_{FONF}, K>0');
subplot(2,3,6); imagesc(log(1+abs(F_filt))); axis image off; title('(f) |F| notched');
% saveas(gcf,'fig6_stripe.png');   % uncomment to export for LaTeX
end

% ------------------------------------------------------------------------
function D = imdilate_local(R)
% 3x3 max-filter without Image Processing Toolbox
D = R;
D = max(D, circshift(R,[1 0]));  D = max(D, circshift(R,[-1 0]));
D = max(D, circshift(R,[0 1]));  D = max(D, circshift(R,[0 -1]));
D = max(D, circshift(R,[1 1]));  D = max(D, circshift(R,[-1 -1]));
D = max(D, circshift(R,[1 -1])); D = max(D, circshift(R,[-1 1]));
end

function h = estimate_hwhm(R, py, px, rho)
% crude radial HWHM around a peak in normalized-frequency units
pk   = R(py,px);
half = pk/2;
[Ny,Nx] = size(R);
rr = 1; found = false;
while rr < 8 && ~found
    ys = max(1,py-rr):min(Ny,py+rr);
    xs = max(1,px-rr):min(Nx,px+rr);
    patch = R(ys,xs);
    if any(patch(:) < half), found = true; else, rr = rr+1; end
end
% convert pixel radius rr to normalized frequency width
h = rr * (1/max(Nx,Ny));
h = max(h, 0.01);
end

function p = psnr_local(x, ref)
mse = mean((x(:)-ref(:)).^2);
peak = max(ref(:));
p = 10*log10(peak^2 / mse);
end
