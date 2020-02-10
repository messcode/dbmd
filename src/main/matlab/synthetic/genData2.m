function simData = genData2(l, n, r, c, opts)
%GENDATA generate the simulation data
%  H are drwan from Dirichlet distirbution with opts.alpha
%opts options contain the following 
%     .coh: coherence
%     .a: a scalar
%     .l:   number of non-zero entriEcs in column of basis matrix
%     .sigmas:  Gaussian noise variance
%     .alpha:  parameters of Dirichlet distritbution
m = l * r - (r - 1) * opts.coh;
simData.W = zeros(m, r);
for i = 1:r
    st = 1 + (i - 1) * l - (i - 1) * opts.coh;
    simData.W(st:st+l-1, i) = 1;
end
simData.W = simData.W * opts.a;
% generate H
assert(numel(opts.alpha ) == r);
if isfield(opts, 'Hcs')
    simData.Hcs = opts.Hcs;
else
    simData.Hcs = cell(c, 1);
    for i = 1:c
        H = drchrnd(opts.alpha, n);
        simData.Hcs{i} = H'; 
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

