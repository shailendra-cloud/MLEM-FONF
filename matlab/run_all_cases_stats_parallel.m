%% ================= run_all_cases_stats_parallel.m =================
% Parallel batch evaluation over multiple noise realizations using parfor.
%
% Seeds are independent, so they are distributed across CPU cores. This
% gives ~ (number of workers)x speedup with the SAME tested CPU kernels.
%
% Output:
%   * per-case mean +/- std printed to Command Window
%   * results_stats.csv (long format) and results_stats.mat
%
% NOTES
%   * Noise for all seeds is pre-generated OUTSIDE parfor for reproducibility
%     (parfor workers do not share the base RNG state).
%   * Same hyper-parameters across all seeds within each case.
%   * FuzzyAD uses whatever is set inside MLEM_FuzzyAD.m / anisodiff2D_fuzzy.m
%     (niter = 3, sigLow = 0.5, as used for the reported results).

clc; clear; close all force;

%% ----------------------------- CONFIG --------------------------------
N_SEEDS      = 10;
phantom_size = 256;
angle        = 0:4:359;
iter_MLEM    = 1000;
USE_POISSON  = true;
INCLUDE_THORAX = true;

% Start / size the parallel pool (uses all physical cores by default).
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool;                      % default cluster, all cores
end
fprintf('Parallel pool: %d workers\n', pool.NumWorkers);

% Method labels (handles built inside the worker loop to avoid broadcast issues)
mLabels = {'MLEM','MLEM+TV','MLEM+AD','MLEM+FuzzyAD','MLEM+FONF','PnP-ADMM(TV)'};
nM = numel(mLabels);
metricNames = {'SNR','MSE','RMSE','PSNR','CP','MSSIM'}; nMet = numel(metricNames);

cases = {
    'TC1_SheppLogan', 1, 0.005, 1/300, 1/7
    'TC2',            2, 0.05,  1/100, 1/10
    'TC3',            3, 0.05,  1/250, 1/7
};

rawAll = {}; caseNames = {};

%% ------------------------ SYNTHETIC CASES ----------------------------
for c = 1:size(cases,1)
    cname = cases{c,1}; job = cases{c,2};
    Pth = cases{c,3}; kappa = cases{c,4}; delta_t = cases{c,5};
    fprintf('\n############ CASE %s ############\n', cname);

    [Weights, Projections, image_vec] = oshotomo1(phantom_size, angle, 150, job);
    gt = reshape(image_vec, [phantom_size, phantom_size]);

    raw = run_case_par(Weights, Projections, gt, kappa, iter_MLEM, delta_t, ...
                       N_SEEDS, USE_POISSON, Pth, nM, nMet);
    rawAll{end+1} = raw; caseNames{end+1} = cname; %#ok<SAGROW>
end

%% --------------------------- REAL THORAX -----------------------------
if INCLUDE_THORAX && exist('thoraxphantom.tif','file')
    fprintf('\n############ CASE Thorax (real) ############\n');
    Pth = 0.05; kappa = 5; delta_t = 0.25;
    im1 = rgb2gray(imread('thoraxphantom.tif'));
    im  = double(imresize(im1,[phantom_size phantom_size],'bicubic'));
    im  = im / max(im(:)); gt = im;
    [Weights, Projections, ~, ~] = buildWeightMatrix(im, angle, 'simple');
    raw = run_case_par(Weights, Projections, gt, kappa, iter_MLEM, delta_t, ...
                       N_SEEDS, USE_POISSON, Pth, nM, nMet);
    rawAll{end+1} = raw; caseNames{end+1} = 'Thorax_real';
elseif INCLUDE_THORAX
    warning('thoraxphantom.tif not found; skipping thorax.');
end

%% ------------------- PRINT & EXPORT MEAN +/- STD ---------------------
fid = fopen('results_stats.csv','w');
fprintf(fid,'Case,Method,Metric,Mean,Std,N\n');
for c = 1:numel(rawAll)
    raw = rawAll{c}; mu = mean(raw,3); sd = std(raw,0,3);
    fprintf('\n===== %s : mean +/- std over %d seeds =====\n', caseNames{c}, N_SEEDS);
    fprintf('%-14s','Metric'); for k=1:nM, fprintf('%22s',mLabels{k}); end; fprintf('\n');
    for j = 1:nMet
        fprintf('%-14s', metricNames{j});
        for k = 1:nM
            fprintf('%12.4f +/-%6.4f', mu(k,j), sd(k,j));
            fprintf(fid,'%s,%s,%s,%.6f,%.6f,%d\n', ...
                caseNames{c}, mLabels{k}, metricNames{j}, mu(k,j), sd(k,j), N_SEEDS);
        end
        fprintf('\n');
    end
end
fclose(fid);
save('results_stats.mat','rawAll','caseNames','mLabels','metricNames','N_SEEDS');
fprintf('\nSaved results_stats.csv and results_stats.mat\n');

%% ===================== local helpers ========================
function raw = run_case_par(W, Proj, gt, kappa, iterN, dt, ...
                            N_SEEDS, usePoisson, Pth, nM, nMet)
    % Pre-generate noise for every seed OUTSIDE parfor (reproducible).
    noiseStack = cell(N_SEEDS,1);
    for s = 1:N_SEEDS
        rng(s);
        noiseStack{s} = add_noise(Proj, Pth, usePoisson);
    end

    raw = zeros(nM, nMet, N_SEEDS);
    parfor s = 1:N_SEEDS
        Pn = noiseStack{s};
        raw(:,:,s) = run_all_methods(W, Pn, gt, kappa, iterN, dt, nM, nMet);
        fprintf('  seed %d done\n', s);
    end
end

function row = run_all_methods(W, Pn, gt, kappa, iterN, dt, nM, nMet)
    row = zeros(nM, nMet);
    % order MUST match mLabels: MLEM, MLEM+TV, MLEM+AD, MLEM+FuzzyAD, FONF, PnP
    [~,m1,r1,s1,p1,c1,~,ss1,it1,~] = MLEM_o3     (W,Pn,gt,kappa,iterN,dt,0,0);
    [~,m2,r2,s2,p2,c2,~,ss2,it2,~] = MLEM_TV     (W,Pn,gt,kappa,iterN,dt,1,0);
    [~,m3,r3,s3,p3,c3,~,ss3,it3,~] = MLEM_o3     (W,Pn,gt,kappa,iterN,dt,1,0);
    [~,m4,r4,s4,p4,c4,~,ss4,it4,~] = MLEM_FuzzyAD(W,Pn,gt,kappa,iterN,dt,1,0);
    [~,m5,r5,s5,p5,c5,~,ss5,it5,~] = MLEM_FONF   (W,Pn,gt,kappa,iterN,dt,1,0);
    [~,m6,r6,s6,p6,c6,~,ss6,it6,~] = PnP_ADMM_TV (W,Pn,gt,kappa,iterN,dt,1,0);

    M = {m1,m2,m3,m4,m5,m6}; R = {r1,r2,r3,r4,r5,r6}; S = {s1,s2,s3,s4,s5,s6};
    P = {p1,p2,p3,p4,p5,p6}; C = {c1,c2,c3,c4,c5,c6}; SS = {ss1,ss2,ss3,ss4,ss5,ss6};
    IT = [it1,it2,it3,it4,it5,it6];
    for k = 1:nM
        idx = IT(k)+1;
        % [SNR, MSE, RMSE, PSNR, CP, MSSIM]
        row(k,:) = [S{k}(idx), M{k}(idx), R{k}(idx), P{k}(idx), C{k}(idx), SS{k}(idx)];
    end
end

function Pn = add_noise(Proj, Pth, usePoisson)
    if usePoisson
        scale = 1/max(Pth,eps); counts = max(Proj*scale,0);
        Pn = poissrnd(counts)/scale;
    else
        noise = rand(size(Proj)); noise = noise/norm(noise);
        Pn = Proj + Pth*norm(Proj)*noise;
    end
end
