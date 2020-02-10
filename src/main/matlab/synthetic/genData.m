function simData = genData(l, n, r, c, opts)
%GENDATA generate the simulation data

%opts options contain the following 
%     .coh: coherence
%     .a: a scalar
%     .l:   number of non-zero entriEcs in column of basis matrix
%     .sigmas:  Gaussian noise variance
%     .p:  bernoulle distribution
m = l * r - (r - 1) * opts.coh;
simData.W = zeros(m, r);
for i = 1:r
    st = 1 + (i - 1) * l - (i - 1) * opts.coh;
    simData.W(st:st+l-1, i) = 1;
end
simData.W = simData.W * opts.a;
% generate H
if isfield(opts, 'Hcs')
    simData.Hcs = opts.Hcs;
else
    simData.Hcs = cell(c, 1);
    for i = 1:c
        H = zeros(r, n);
        H(1:r, :) = rand(r, n) < opts.p;
        H(end, sum(H) == 0) = 1;
        simData.Hcs{i} = bsxfun(@rdivide, H, sum(H)); 
    end
end
% generate Xc
simData.Xcs = cell(c,  1);
simData.Ecs = cell(c, 1);
for i = 1:c
    simData.Ecs{i} = randn(m, n) * opts.sigmas(i);
    simData.Xcs{i} = simData.W * simData.Hcs{i} + simData.Ecs{i};
end
end

