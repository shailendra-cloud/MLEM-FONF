%% ablation_alpha.m  —  Integer- vs fractional-order ablation for MLEM+FONF
% -------------------------------------------------------------------------
% Uses the SAME pipeline as alpha_sensitivity.m (oshotomo1 + MLEM_FONF), so
% the numbers are directly comparable to Table II / Fig. 4.
% Only difference: we evaluate alpha = 0.5 (proposed, fractional) against
% alpha = 1.0 (integer-order baseline), with mean +/- std over noise seeds.
% Run from the folder that contains oshotomo1.m and MLEM_FONF.m
% -------------------------------------------------------------------------
clc; clear; close all force;

alphas       = [0.5, 1.0];   % 0.5 = proposed (fractional) ; 1.0 = integer-order
% (optional: add 2.0 as a second integer reference -> alphas = [0.5 1.0 2.0])
N_SEEDS      = 10;           % paper reports mean over 10 noise realizations
phantom_size = 256;
angle        = 0:4:359;
iter_MLEM    = 1000;
job = 1; Pth = 0.005; kappa = 1/300; delta_t = 1/7;

% ---- build system matrix, projections, ground truth (same as the driver) ----
[Weights, Projections, image_vec] = oshotomo1(phantom_size, angle, 150, job);
gt = reshape(image_vec, [phantom_size, phantom_size]);

nA = numel(alphas);
psnr_mean = zeros(nA,1); psnr_std = zeros(nA,1);
snr_mean  = zeros(nA,1); snr_std  = zeros(nA,1);

for a = 1:nA
    psnr_runs = zeros(N_SEEDS,1); snr_runs = zeros(N_SEEDS,1);
    fp = struct('alpha', alphas(a), 'lambda_frac', 0.5, ...
                'notch_freqs', [], 'notch_widths', [], 'beta_spec', 0.6);
    for s = 1:N_SEEDS
        rng(s);
        scale  = 1/max(Pth,eps);
        counts = max(Projections*scale, 0);
        Pn     = poissrnd(counts)/scale;
        [~,~,~,snr,psnr,~,~,~,it,~] = ...
            MLEM_FONF(Weights, Pn, gt, kappa, iter_MLEM, delta_t, 1, 0, fp);
        psnr_runs(s) = psnr(it+1);
        snr_runs(s)  = snr(it+1);
    end
    psnr_mean(a)=mean(psnr_runs); psnr_std(a)=std(psnr_runs);
    snr_mean(a) =mean(snr_runs);  snr_std(a) =std(snr_runs);

    if     abs(alphas(a)-0.5)<1e-9, tag='fractional (proposed)';
    elseif abs(alphas(a)-1.0)<1e-9, tag='integer-order        ';
    else,                            tag='                     '; end
    fprintf('alpha=%.2f [%s]: PSNR=%.2f +/- %.2f   SNR=%.2f +/- %.2f\n', ...
        alphas(a), tag, psnr_mean(a), psnr_std(a), snr_mean(a), snr_std(a));
end

%% ---- comparison + LaTeX-ready rows ----
iF = find(abs(alphas-0.5)<1e-9, 1);    % fractional (proposed)
iI = find(abs(alphas-1.0)<1e-9, 1);    % integer-order
if ~isempty(iF) && ~isempty(iI)
    fprintf('\nPSNR(alpha=0.5) - PSNR(alpha=1.0) = %+.2f dB\n', psnr_mean(iF)-psnr_mean(iI));
    fprintf('SNR (alpha=0.5) - SNR (alpha=1.0) = %+.2f dB\n', snr_mean(iF)-snr_mean(iI));
    fprintf('\n%% ---- LaTeX rows (SNR | PSNR, mean +/- std) ----\n');
    fprintf('Integer-order ($\\alpha{=}1$)          & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \\\\\n', ...
        snr_mean(iI), snr_std(iI), psnr_mean(iI), psnr_std(iI));
    fprintf('Fractional ($\\alpha{=}0.5$, proposed) & %.2f $\\pm$ %.2f & %.2f $\\pm$ %.2f \\\\\n', ...
        snr_mean(iF), snr_std(iF), psnr_mean(iF), psnr_std(iF));
end

%% ---- save csv ----
fid = fopen('ablation_alpha.csv','w');
fprintf(fid,'alpha,PSNR_mean,PSNR_std,SNR_mean,SNR_std,N\n');
for a = 1:nA
    fprintf(fid,'%.2f,%.6f,%.6f,%.6f,%.6f,%d\n', ...
        alphas(a), psnr_mean(a), psnr_std(a), snr_mean(a), snr_std(a), N_SEEDS);
end
fclose(fid);
fprintf('\nSaved ablation_alpha.csv\n');

% Note on interpretation:
%  PSNR rises monotonically with alpha over [0.2, 0.9] (paper, Fig. 4a), so
%  alpha = 1.0 typically yields a higher PSNR than alpha = 0.5. The paper
%  fixes alpha = 0.5 as a conservative noise-vs-detail balance, not as a
%  PSNR maximizer.