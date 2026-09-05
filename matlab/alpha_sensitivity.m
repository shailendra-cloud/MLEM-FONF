%% alpha_sensitivity.m
clc; clear; close all force;

alphas       = 0.2:0.1:0.9;
N_SEEDS      = 5;
phantom_size = 256;
angle        = 0:4:359;
iter_MLEM    = 1000;
job = 1; Pth = 0.005; kappa = 1/300; delta_t = 1/7;

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
        scale = 1/max(Pth,eps); counts = max(Projections*scale,0);
        Pn = poissrnd(counts)/scale;
        [~,~,~,snr,psnr,~,~,~,it,~] = ...
            MLEM_FONF(Weights, Pn, gt, kappa, iter_MLEM, delta_t, 1, 0, fp);
        psnr_runs(s) = psnr(it+1);
        snr_runs(s)  = snr(it+1);
    end
    psnr_mean(a)=mean(psnr_runs); psnr_std(a)=std(psnr_runs);
    snr_mean(a)=mean(snr_runs);   snr_std(a)=std(snr_runs);
    fprintf('alpha=%.2f : PSNR=%.2f +/- %.2f   SNR=%.2f +/- %.2f\n', ...
        alphas(a), psnr_mean(a), psnr_std(a), snr_mean(a), snr_std(a));
end

figure('Color','w');
errorbar(alphas, psnr_mean, psnr_std, '-o', 'LineWidth',1.4, ...
    'MarkerFaceColor',[0.90 0.10 0.75], 'Color',[0.90 0.10 0.75]);
xlabel('Fractional order \alpha'); ylabel('PSNR (dB)');
title('Sensitivity of MLEM+FONF to \alpha (mean \pm std)');
grid on; box on;

fid = fopen('alpha_sensitivity.csv','w');
fprintf(fid,'alpha,PSNR_mean,PSNR_std,SNR_mean,SNR_std,N\n');
for a=1:nA
    fprintf(fid,'%.2f,%.6f,%.6f,%.6f,%.6f,%d\n', ...
        alphas(a),psnr_mean(a),psnr_std(a),snr_mean(a),snr_std(a),N_SEEDS);
end
fclose(fid);
fprintf('\nSaved alpha_sensitivity.csv\n');