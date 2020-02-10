function [Wcs, Hcs, W, out] = admmAls(Xcs, Hcs,  maxIter, opts)
% ADMM_ALS funtion for solve th update_W and update_Wc.
% 
%Input
%  Xs:  cell of data m * n_c matrices
%  Wcs:  cell of Wc   m * r
%  Hcs:  cell of Hc   r * n_c, which is known
%  opt: options of iterations. It conatains following fields
%     .noise_aware: boolean wehter update
%     .rho: ADMM hyperparameters
%     .lam: L1 penalty parameter
%     
C = length(Xcs);
[m, n] = size(Xcs{1});
[r, ~] = size(Hcs{1});
%hasTrueW = isfield(opts, 'trueW');
nonSmooth = isfield(opts, 'lam');
if nonSmooth
    lam = opts.lam;
    assert(opts.lam > 0);
    fprintf('NonSmooth lam=%.2f\n', lam)
end
Ucs = cell(C, 1);
sigmas = ones(C, 1);
out.viol = zeros(maxIter, 1);
out.loss = zeros(maxIter, 1);
Ucs(:) = {0};
rho = opts.rho;
% initialize W
W = zeros(m, r);
rows = randsample(1:n * C, r, false);
for idx = 1:r
    c = ceil(rows(r) / n);
    if rem(rows(r), n) == 0
        W(:, r) = Xcs{c}(:, n);
    else
        W(:, r) = Xcs{c}(:, rem(rows(r), n));
    end
end
% initialize Wc
Wcs = cell(C, 1);
for c = 1:C
    Wcs{c} = W;
end


for nIter = 1 : maxIter
    % track information
    % compute residual
    loss = 0;
    for c = 1:C
     loss = loss + sum(sum((Xcs{c} - W * Hcs{c}).^2));
    end
    if nonSmooth
     loss = loss + lam * sum(abs(W(:)));
    end       
    out.loss(nIter) = loss / C;    
    % compute viol
    viol = 0;
    for c = 1:C
        viol = viol + norm(Wcs{c} - W, 'fro');
    end
    out.viol(nIter) = viol / C;    
    % update Wc
    for c = 1:C
        Hc = Hcs{c};
        Wcs{c} = (Xcs{c} * Hc' / rho + W  - Ucs{c}) / (eye(r) + Hc * Hc' / rho);
    end
    % update W
    W = 0;
    sigmasRs = sum(1 ./ sigmas);
    for c = 1:C
        W = W +  1 / (sigmas(c) * sigmasRs) * (Wcs{c} + Ucs{c});
    end
    if nonSmooth
        W = softhreolding(W, lam / (C * rho));
    end
    % update Uc
    for c = 1:C
        Ucs{c} = Ucs{c} + Wcs{c} - W;
    end
    % update sigmas
    if opts.noise_aware
        sigmas = calSimgas(Xcs, Wcs, Hcs);
    end
end 
out.sigmas = sigmas;
end
