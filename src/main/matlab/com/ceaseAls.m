function  [Wcs, Hcs, W, out] = ceaseAls(Xcs, Hcs,  maxIter, opts)
% 
%Input
%  Xs:  cell of data m * n_c matrices
%  Wcs:  cell of Wc   m * r
%  Hcs:  cell of Hc   r * n_c, which is known
%  opt: options of iterations. It conatains following fields
%     .a: proximal point paramter recomendation  0.15p/n 
%     .noise_aware: boolean matrix
%     .lam: l_1 reuglarization parameters
%Output
%   out:
%      .loss 
C = length(Xcs);
[m, n] = size(Xcs{1});
[r, ~] = size(Hcs{1});
a = opts.a;
sigmas = ones(C, 1);
hasTrueW = isfield(opts, 'trueW');
nonSmooth = isfield(opts, 'lam');
if nonSmooth
    lam = opts.lam;
    assert(lam > 0);
    fprintf('NonSmooth lam=%.2f\n', lam)
end
% sigmas = ones(C, 1);

out.loss = zeros(maxIter, 1);
if hasTrueW
    out.diff = zeros(maxIter, 1);
end
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

for nIter = 1:maxIter
   % track information
   % compute the losss
   loss = 0;
   for c = 1:C
    loss = loss + sum(sum((Xcs{c} - W * Hcs{c}).^2));
   end
   if nonSmooth
     loss = loss + lam * sum(abs(W(:)));
   end   
   out.loss(nIter) = loss / C;
   
   % compute diff
   if hasTrueW
       out.diff(nIter) = norm(opts.trueW - W, 2); 
   end
   sigmasRs = sum(1 ./ sigmas);
   % compute gradient on central processor
   gradF = 0;
   for c = 1:C
       gradF = gradF + 1 / (sigmas(c) * sigmasRs) * (W * Hcs{c} - Xcs{c}) * Hcs{c}';
   end
   % update on nodes
   if nonSmooth
       for c = 1:C
          Wcs{c} = fista(Xcs{c}, Hcs{c}, W, lam , a, gradF);
       end
   else
       for c = 1:C
           HHt = Hcs{c} * Hcs{c}';
           Wcs{c} = (W * HHt - gradF + a * W) / (HHt + a * eye(r)); 
       end
   end
   % averaging on central processor
   W = 0;
   for c = 1:C
        W = W +  1 / (sigmas(c) * sigmasRs) * Wcs{c};
   end
   % update sigmas
   if opts.noise_aware
       sigmas = calSimgas(Xcs, Wcs, Hcs);
   end
end
out.sigmas = sigmas;
end

