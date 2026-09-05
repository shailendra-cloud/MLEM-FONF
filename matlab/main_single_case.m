%% ================= main_single_case.m =================
% Interactive single-case comparison of all six methods on one dataset
% and one noise realization (the batch driver with statistics is
% run_all_cases_stats_parallel.m).
%
%   * Methods: MLEM, MLEM+TV, MLEM+AD, MLEM+FuzzyAD, MLEM+FONF (proposed,
%     Algorithm 1 via MLEM_FONF.m), and the training-free PnP-ADMM(TV).
%   * Poisson noise matches the measurement model of the paper.
%   * Geometry: angle = 0:4:359 gives 90 projection angles; oshotomo1
%     (AIR Tools' paralleltomo) uses p parallel rays per angle.

clc; clear; close all force;

%% ----------------------- DATASET SELECTION ---------------------------
DataSet = menu('Select a dataset:', '1. Synthetic Phantom', '2. CT/PET Images');

phantom_size = 256;
angle        = 0:4:359;     % 90 projection angles
iter_MLEM    = 1000;

if DataSet == 1
    DataSetS = menu('Select a Testcase:', ...
        '1. Modified Shepp-Logan', '2. Testcase2', '3. Testcase3');
    switch DataSetS
        case 1, Pth = 0.005; job = 1; kappa = 1/300; delta_t = 1/7;
        case 2, Pth = 0.05;  job = 2; kappa = 1/100; delta_t = 1/10;
        otherwise, Pth = 0.05; job = 3; kappa = 1/250; delta_t = 1/7;
    end
    [Weights, Projections, image_vec] = oshotomo1(phantom_size, angle, 150, job);
    image_gt2d = reshape(image_vec, [phantom_size, phantom_size]);
else
    Pth = 0.05; kappa = 5; delta_t = 0.25;
    im1 = rgb2gray(imread('thoraxphantom.tif'));
    im  = double(imresize(im1, [phantom_size phantom_size], 'bicubic'));
    im  = im / max(im(:));                 % normalise to [0,1]
    image_gt2d = im;
    [Weights, Projections, ~, ~] = buildWeightMatrix(im, angle, 'simple');
end

%% --------------------------- ADD NOISE -------------------------------
NoiseModel = menu('Noise model:', '1. Poisson (matches paper)', '2. Additive (legacy)');
if NoiseModel == 1
    scale  = 1 / max(Pth, eps);            % higher scale -> lower relative noise
    counts = max(Projections * scale, 0);
    Proj_noise = poissrnd(counts) / scale;
else
    noise = rand(size(Projections));
    noise = noise / norm(noise);
    Proj_noise = Projections + Pth * norm(Projections) * noise;
end

%% ====================== RUN ALL METHODS (honest) =====================
labels = {'MLEM','MLEM+TV','MLEM+AD','MLEM+FuzzyAD','MLEM+FONF','PnP-ADMM(TV)'};
% Paper Fig. 3 panels (a)-(f): Original + the five MLEM-family methods;
% PnP-ADMM(TV) is the sixth compared method of Table II.
nM     = numel(labels);
X      = cell(1,nM);
SNRc   = zeros(1,nM); MSEc = zeros(1,nM); RMSEc = zeros(1,nM);
PSNRc  = zeros(1,nM); CPc  = zeros(1,nM); MSSIMc = zeros(1,nM);
times  = zeros(1,nM);
curves = struct('psnr',cell(1,nM),'snr',[],'rmse',[],'mssim',[],'it',[]);

% Each method is called with its OWN correct function; results are stored
% inline against the matching label. (No nested function: MATLAB scripts do
% not share workspace with local functions.)

for k = 1:nM
    tStart = tic;
    switch k
        case 1, disp('--- MLEM ---');
            [x,mse,rmse,snr,psnr,cp,~,mssim,it,~] = ...
                MLEM_o3(Weights,Proj_noise,image_gt2d,kappa,iter_MLEM,delta_t,0,0);
        case 2, disp('--- MLEM+TV ---');
            [x,mse,rmse,snr,psnr,cp,~,mssim,it,~] = ...
                MLEM_TV(Weights,Proj_noise,image_gt2d,kappa,iter_MLEM,delta_t,1,0);
        case 3, disp('--- MLEM+AD ---');
            [x,mse,rmse,snr,psnr,cp,~,mssim,it,~] = ...
                MLEM_o3(Weights,Proj_noise,image_gt2d,kappa,iter_MLEM,delta_t,1,0);
        case 4, disp('--- MLEM+FuzzyAD ---');
            [x,mse,rmse,snr,psnr,cp,~,mssim,it,~] = ...
                MLEM_FuzzyAD(Weights,Proj_noise,image_gt2d,kappa,iter_MLEM,delta_t,1,0);
        case 5, disp('--- MLEM+FONF (proposed) ---');   % Algorithm 1
            [x,mse,rmse,snr,psnr,cp,~,mssim,it,~] = ...
                MLEM_FONF(Weights,Proj_noise,image_gt2d,kappa,iter_MLEM,delta_t,1,0);
        case 6, disp('--- PnP-ADMM(TV) [extra baseline, Comment 6] ---');
            [x,mse,rmse,snr,psnr,cp,~,mssim,it,~] = ...
                PnP_ADMM_TV(Weights,Proj_noise,image_gt2d,kappa,iter_MLEM,delta_t,1,0);
    end
    times(k) = toc(tStart);

    X{k}     = reshape(x, [phantom_size phantom_size]);
    idx      = it + 1;
    SNRc(k)  = snr(idx);  MSEc(k)  = mse(idx);  RMSEc(k) = rmse(idx);
    PSNRc(k) = psnr(idx); CPc(k)   = cp(idx);   MSSIMc(k)= mssim(idx);
    curves(k).psnr  = psnr; curves(k).snr = snr; curves(k).rmse = rmse;
    curves(k).mssim = mssim; curves(k).it = it;
    fprintf('Time %s: %.3f s\n', labels{k}, times(k));
end

%% --------------------- QUALITATIVE RESULTS ----------------------
figure;
subplot(2,4,1), imshow(mat2gray(image_gt2d)), title('Original');
for k = 1:nM
    subplot(2,4,k+1), imshow(mat2gray(X{k})), title(labels{k});
end

%% ----------------------- METRIC CURVES -------------------------
step = 20;
plot_metric = @(field, ylab) plot_curves(curves, labels, field, ylab, step);
plot_metric('psnr','PSNR (dB)');
plot_metric('snr','SNR (dB)');
plot_metric('rmse','RMSE');
plot_metric('mssim','MSSIM');

%% ----------------------- SUMMARY TABLE -------------------------
tableData = [SNRc; MSEc; RMSEc; PSNRc; CPc; MSSIMc];
figure('Position',[150 150 900 250]);
uitable('Data',tableData, 'ColumnName',labels, ...
        'RowName',{'SNR','MSE','RMSE','PSNR','CP','MSSIM'}, ...
        'Position',[20 30 860 200]);

fprintf('\n=========== RESULTS (convergence) ===========\n');
fprintf('%-14s %8s %10s %8s %8s\n','Method','SNR','PSNR','CP','MSSIM');
for k = 1:nM
    fprintf('%-14s %8.2f %10.2f %8.3f %8.3f   (%.1fs)\n', ...
        labels{k}, SNRc(k), PSNRc(k), CPc(k), MSSIMc(k), times(k));
end

%% ----------------------- LINE PROFILE (with zoom inset) ---------
row = round(phantom_size/2);
[clr, mk] = method_styles();          % distinct colours + markers

figLP = figure('Name','Line profile (mid-row)','Color','w'); hold on;
plot(image_gt2d(row,:), 'k-', 'LineWidth', 1.4);                 % ground truth
for k = 1:nM
    plot(X{k}(row,:), '-', 'Color', clr(k,:), 'Marker', mk{k}, ...
         'MarkerSize', 3, 'MarkerIndices', 1:8:phantom_size, 'LineWidth', 0.8);
end
xlabel('Pixel position'); ylabel('Intensity');
legend(['Original', labels], 'Location','NorthEastOutside');
grid on; box on; hold off;

% ---- pick a zoom window automatically around the strongest edge ----
gt = image_gt2d(row,:);
[~, edgePix] = max(abs(diff(gt)));            % strongest intensity transition
zHalf = 30;                                   % half-width of zoom window
zlo = max(1, edgePix - zHalf);
zhi = min(phantom_size, edgePix + zHalf);

% ---- inset axes inside the same figure ----
axInset = axes('Parent', figLP, 'Position', [0.18 0.55 0.30 0.32]); hold(axInset,'on');
plot(axInset, zlo:zhi, gt(zlo:zhi), 'k-', 'LineWidth', 1.4);
for k = 1:nM
    plot(axInset, zlo:zhi, X{k}(row, zlo:zhi), '-', 'Color', clr(k,:), ...
         'Marker', mk{k}, 'MarkerSize', 3, 'LineWidth', 0.8);
end
grid(axInset,'on'); box(axInset,'on');
title(axInset, sprintf('Zoom: pixels %d-%d', zlo, zhi), 'FontSize', 8);
xlim(axInset, [zlo zhi]);

% ---- mark the zoom region on the main plot ----
yl = ylim;
plot([zlo zlo], yl, 'k:', 'LineWidth', 0.6);
plot([zhi zhi], yl, 'k:', 'LineWidth', 0.6);

%% =================== local helpers ====================
function plot_curves(curves, labels, field, ylab, step)
    [clr, mk] = method_styles();
    figure('Color','w'); hold on;
    for k = 1:numel(labels)
        y  = curves(k).(field);
        it = curves(k).it + 1;
        idx = 1:step:it;
        plot(idx, y(idx), '-', 'Color', clr(k,:), 'Marker', mk{k}, ...
             'MarkerSize', 4, 'LineWidth', 1.0);
    end
    xlabel('Iteration'); ylabel(ylab);
    legend(labels, 'Location','NorthEastOutside');
    grid on; box on; hold off;
end

function [clr, mk] = method_styles()
    % Six visually distinct colours (no two methods share a colour) + markers
    % Order matches: MLEM, MLEM+TV, MLEM+AD, MLEM+FuzzyAD, MLEM+FONF, PnP-ADMM(TV)
    clr = [ 0.20 0.65 0.20;    % MLEM         - green
            0.00 0.75 0.85;    % MLEM+TV      - cyan
            0.85 0.20 0.20;    % MLEM+AD      - red
            0.30 0.30 0.90;    % MLEM+FuzzyAD - blue
            0.90 0.10 0.75;    % MLEM+FONF    - magenta/pink (proposed)
            0.95 0.55 0.10 ];  % PnP-ADMM(TV) - orange
    mk = {'d','p','o','s','^','v'};
end
