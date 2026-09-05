function H = fonf_eq4(N, alpha, notches)
% FONF_EQ4  H_FONF exactly as printed in Eq. (4) of the paper.
%
%   H(w) = G_alpha(w) * prod_m [1 - exp(-||w - w_m||^2/(2 sigma_m^2))]^(alpha/2)
%   G_alpha(w) = exp(-alpha ||w||^2 / (4 w_N^2)),  w_N = pi (Nyquist radius).
%
%   N       : image size (N x N), returns H on the unshifted FFT grid
%   alpha   : fractional order in (0, 2]
%   notches : K x 3 rows [wx_m, wy_m, sigma_m] in rad/pixel (conjugates
%             are applied automatically). [] gives the K = 0 base G_alpha.
%
%   Note: the as-run pipeline (fonf_filter2D_improved.m) uses a rational
%   fractional-order base; this function provides the closed form of
%   Eq. (4) for reference and for the K > 0 notch experiments.

  if nargin < 3, notches = []; end
  f  = [0:ceil(N/2)-1, -floor(N/2):-1] / N;     % cycles/pixel (FFT order)
  [WX, WY] = meshgrid(2*pi*f, 2*pi*f);
  WR2  = WX.^2 + WY.^2;
  w_N  = pi;
  H = exp(-alpha * WR2 / (4 * w_N^2));
  for m = 1:size(notches, 1)
    cx = notches(m,1); cy = notches(m,2); sg = notches(m,3);
    for s = [1, -1]
      d2 = (WX - s*cx).^2 + (WY - s*cy).^2;
      H  = H .* (1 - exp(-d2 / (2*sg^2))).^(alpha/2);
    end
  end
end
