function psnr_Value = PSNR(A, B)
% PSNR  Peak Signal-to-Noise Ratio between reference A and test B.
%
%   The peak value is derived from the reference signal, so the measure is
%   correct whether the images are in [0,1] or [0,255].

    if ~isequal(size(A), size(B))
        error('PSNR: the two inputs have different sizes.');
    end

    A = double(A);
    B = double(B);

    mse = mean((A(:) - B(:)).^2);

    if mse == 0
        disp('Images are identical: PSNR is infinite.');
        psnr_Value = Inf;
        return;
    end

    peak = max(A(:));          % true peak of the reference (~1 for [0,1] data)
    if peak <= 0
        error('PSNR: reference peak is non-positive; check input scaling.');
    end

    psnr_Value = 10 * log10(peak^2 / mse);
end
