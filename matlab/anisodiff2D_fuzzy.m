function diff_im = anisodiff2D_fuzzy(im, num_iter, delta_t, kappa, ~)
%ANISODIFF2D_FUZZY  Fuzzy nonlinear anisotropic diffusion.
%
%   A genuine fuzzy variant of Perona-Malik diffusion. The conduction
%   coefficient is obtained by FUZZY INFERENCE on the local gradient
%   magnitude instead of a single fixed Perona-Malik function:
%
%     - Fuzzy set LOW gradient  (flat region)  -> diffuse strongly
%     - Fuzzy set HIGH gradient (edge/texture) -> diffuse weakly
%
%   Gaussian membership functions are evaluated for each of the 8 neighbour
%   directions, and the conduction is defuzzified by a weighted average.
%   This preserves edges (low conduction where gradient is high) while
%   smoothing flat regions, and is smoother/more robust than crisp AD.
%
%   Concept follows fuzzy AD penalties for emission tomography
%   (Zhu et al., Med. Biol. Eng. Comput., 2006).
%
%   im, num_iter, delta_t, kappa : as in anisodiff2D. Last arg ignored.

    im = double(im);
    diff_im = im;

    dx = 1; dy = 1; dd = sqrt(2);

    hN = [0 1 0; 0 -1 0; 0 0 0];
    hS = [0 0 0; 0 -1 0; 0 1 0];
    hE = [0 0 0; 0 -1 1; 0 0 0];
    hW = [0 0 0; 1 -1 0; 0 0 0];
    hNE = [0 0 1; 0 -1 0; 0 0 0];
    hSE = [0 0 0; 0 -1 0; 0 0 1];
    hSW = [0 0 0; 0 -1 0; 1 0 0];
    hNW = [1 0 0; 0 -1 0; 0 0 0];

    % fuzzy membership spreads (relative to kappa)
    sigLow  = 0.5;     % "low gradient" membership width
    sigHigh = 1.0;     % "high gradient" membership width

    fuzzy_c = @(nabla) fuzzy_conduction(nabla, kappa, sigLow, sigHigh);

    for t = 1:num_iter
        nablaN  = imfilter(diff_im,hN,'conv');
        nablaS  = imfilter(diff_im,hS,'conv');
        nablaW  = imfilter(diff_im,hW,'conv');
        nablaE  = imfilter(diff_im,hE,'conv');
        nablaNE = imfilter(diff_im,hNE,'conv');
        nablaSE = imfilter(diff_im,hSE,'conv');
        nablaSW = imfilter(diff_im,hSW,'conv');
        nablaNW = imfilter(diff_im,hNW,'conv');

        cN  = fuzzy_c(nablaN);   cS  = fuzzy_c(nablaS);
        cW  = fuzzy_c(nablaW);   cE  = fuzzy_c(nablaE);
        cNE = fuzzy_c(nablaNE);  cSE = fuzzy_c(nablaSE);
        cSW = fuzzy_c(nablaSW);  cNW = fuzzy_c(nablaNW);

        diff_im = diff_im + delta_t*( ...
            (1/dy^2)*cN.*nablaN + (1/dy^2)*cS.*nablaS + ...
            (1/dx^2)*cW.*nablaW + (1/dx^2)*cE.*nablaE + ...
            (1/dd^2)*cNE.*nablaNE + (1/dd^2)*cSE.*nablaSE + ...
            (1/dd^2)*cSW.*nablaSW + (1/dd^2)*cNW.*nablaNW );
    end
end

% ---------------------------------------------------------------
function c = fuzzy_conduction(nabla, kappa, sigLow, sigHigh)
% Fuzzy inference on |nabla|/kappa: blend "diffuse" (1) and "preserve" (~0).
    t = abs(nabla) / max(kappa, eps);
    muLow  = exp(-(t.^2) / (2*sigLow^2));        % membership: low gradient
    muHigh = 1 - exp(-(t.^2) / (2*sigHigh^2));   % membership: high gradient
    % defuzzify (weighted average of rule outputs 1 and 0)
    c = muLow ./ (muLow + muHigh + eps);
end
