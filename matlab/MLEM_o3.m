function [L, mse, rmse, snr, psnr, cp, nmse, mssim, iter, cor ] = ...
    MLEM_o3(G , N_true , image , kappa , iter_max , delta_t, control , input )

% Robust MLEM with optional AD regularization (control=1)

if nargin < 6
    iter_max = 100;
end
if nargin < 5
    iter_max = 100;
    kappa    = 1/300 ;
end

image = double(image);

num_pixels   = numel(image);
phantom_size = round(sqrt(num_pixels));
if phantom_size^2 ~= num_pixels
    error('MLEM_o3: image has %d elements, not a perfect square.', num_pixels);
end

image_vec = reshape(image, [phantom_size^2, 1]);
image1    = reshape(image_vec , [ phantom_size  phantom_size ] );

G = G';
L0 = rand( size( G , 1) , 1 );

if input == 1 && exist('SART_only_o1','file')
    L0 = SART_only_o1(G' , N_true , image_vec , 10  , kappa);
end

G1 = sum(G,2);
L  = L0;
snr_old = 0;
snr_new = 0;

mse   = zeros(iter_max+1,1);
rmse  = zeros(iter_max+1,1);
snr   = zeros(iter_max+1,1);
psnr  = zeros(iter_max+1,1);
cp    = zeros(iter_max+1,1);
nmse  = zeros(iter_max+1,1);
mssim = zeros(iter_max+1,1);
cor   = zeros(iter_max+1,1);

curr_image = reshape(L0 , [ phantom_size  phantom_size ] );
[mse(1),rmse(1),snr(1),psnr(1),cp(1),nmse(1),mssim(1),cor(1)] = ...
    error_analysis(image1, curr_image);

for iter  = 1:iter_max
    
    N_cal  = G' * L;
    N_cal  = max(N_cal, eps);
    N_comp = N_true ./ N_cal;
    N_comp(~isfinite(N_comp)) = 0;          % guard 0/0 or Inf from empty rays
    X      = G * N_comp;
    X_norm = X ./ max(G1, eps);
    X_norm(~isfinite(X_norm)) = 0;          % guard division by zero column sums
    L      = L .* X_norm;
    L      = max(L, 0);                      % enforce non-negativity
    L(~isfinite(L)) = 0;                     % final guard
    
    if control == 1
        L1  = reshape(L , [phantom_size , phantom_size] );
        L1(~isfinite(L1)) = 0;               % clean input to diffusion
        niter = 3;
        L2  = anisodiff2D( L1 , niter ,delta_t ,kappa , 2 );
        L2(~isfinite(L2)) = 0;               % clean diffusion output
        L2  = max(L2, 0);
        L   = reshape( L2 , [phantom_size^2 , 1] );
        snr_new = SNR( image1, L2);
        [mse(iter+1),rmse(iter+1),snr(iter+1),psnr(iter+1),cp(iter+1), ...
         nmse(iter+1),mssim(iter+1),cor(iter+1)] = error_analysis(image1, L2);
    else
        L1 = reshape(L , [phantom_size , phantom_size] );
        snr_new = SNR( image1, L1);
        [mse(iter+1),rmse(iter+1),snr(iter+1),psnr(iter+1),cp(iter+1), ...
         nmse(iter+1),mssim(iter+1),cor(iter+1)] = error_analysis(image1, L1);
    end
    
    snr_old = snr_new; %#ok<NASGU>
end

end
