function [W, Hcs, out] = vanillaAls(Xcs, Hcs,  maxIter, opts)
%VANILLAALS use FISTA to update W.
% opts options structor
%    .lam: l1 regularization
%    .mode "FISTA" or "ISTA"
C = length(Xcs);
[m, n] = size(Xcs{1});
[r, ~] = size(Hcs{1});
out.loss = zeros(maxIter, 1);
lam = opts.lam;

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
HHt = 0;
for c = 1:C
    HHt = HHt + Hcs{c} * Hcs{c}';
end
L = eigs(HHt, 1);

if strcmp(opts.mode, 'FISTA')
    preW = W;
    Y = W;
    t = 1;
    for nIter = 1:maxIter
       % track information
       % compute the losss
       loss = 0;
       for c = 1:C
         loss = loss + sum(sum((Xcs{c} - W * Hcs{c}).^2));
       end
       out.loss(nIter) = loss + lam * sum(abs(W(:)));
       % compute grad
       grad = W * HHt;
       for c = 1:C
         grad = grad - Xcs{c} * Hcs{c}';
       end
       W = Y - grad / L;
       W = softhreolding(W, lam / L);
       numerator = t - 1;
       t = .5 + sqrt(1 + 4 * t * t);
       Y = W + (numerator / t) * (W - preW);
       preW = W;
    end
else
    for nIter = 1:maxIter
       loss = 0;
       for c = 1:C
         loss = loss + sum(sum((Xcs{c} - W * Hcs{c}).^2));
       end
       out.loss(nIter) = loss + lam * sum(abs(W(:)));
       grad = W * HHt;
       for c = 1:C
         grad = grad - Xcs{c} * Hcs{c}';
       end
       W = W - grad / L;
       W = softhreolding(W, lam / L);        
    end
end     
% normalize loss
out.loss = out.loss / C;
end

