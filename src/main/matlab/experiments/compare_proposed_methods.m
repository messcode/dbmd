addpath(genpath('matlab'));
%% Non-smooth Experiments Bernoulli
l = 20; n = 100; r = 20; C = 5; coh = 2;
m = l * r - (r - 1) * coh;
n_v = [100, 500, 5000];
maxIter = 100;
lam = 0.01;
% save loss to matrix
ista_loss = zeros(length(n_v), maxIter);
fista_loss = zeros(length(n_v), maxIter);
admm_loss = zeros(length(n_v), maxIter);
cease_loss = zeros(length(n_v), maxIter);

for idx = 1:length(n_v)
sigmas = [1.0, 1.0, 1.0, 1.0, 1.0];
simOpts = struct('coh', coh, 'a', 1.5, 'p', 1 / (r-1), 'sigmas', sigmas);
simData = genData(l, n_v(idx), r, C, simOpts);
% Vanilla -ista
istaOpts = struct('lam', lam, 'mode', 'ISTA');
[~, ~, ista_out] = vanillaAls(simData.Xcs, simData.Hcs, maxIter,istaOpts);
% Vanilla -fista
fistaOpts = struct('lam', lam, 'mode', 'FISTA');
[~, ~, fista_out] = vanillaAls(simData.Xcs, simData.Hcs, maxIter, fistaOpts);
% ADMM
rho = 50;
admmOpts = struct('noise_aware', 0, 'rho', rho, 'lam', lam);
[Wcs, Hcs, W, out] = admmAls(simData.Xcs, simData.Hcs, maxIter, admmOpts);
% cease
ceaseOpts = struct('a', 0.15 * m / n, 'noise_aware', 0, 'lam', lam);
[Wcs_, Hcs_, W_, out_] = ceaseAls(simData.Xcs, simData.Hcs, maxIter, ceaseOpts);
% save to cease
ista_loss(idx, :) = ista_out.loss;
fista_loss(idx, :) = ista_out.loss;
admm_loss(idx, :) = out.loss;
cease_loss(idx, :) = out_.loss;
end

set_fig('units','inches','width', 8,'height', 2,'font','Times New Roman','fontsize', 10);
for idx = 1:3
subplot(1, 3, idx)
hold on;
%plot(log(ista_loss(idx, :)))
plot(log(fista_loss(idx, :)))
plot(log(admm_loss(idx, :)))
plot(log(cease_loss(idx, :)))
xlim([0, 30])
box on;
legend('FISTA', 'ADMM', 'CEASE')
title(sprintf('Nonsmooth, n=%d', n_v(idx)))  
hold off
end
save('../../output/synthetic_A.mat', 'ista_loss', 'fista_loss', 'admm_loss','cease_loss','maxIter')
% export_fig '../../doc/figs/comparison_vary_n_nonsmooth.png' -r600
%% Non-smooth Experiments Dirichlet
l = 20; n = 100; r = 20; C = 5; coh = 2;
m = l * r - (r - 1) * coh;
n_v = [100, 500, 5000];
maxIter = 100;
lam = 0.01;
% save loss to matrix
ista_loss = zeros(length(n_v), maxIter);
fista_loss = zeros(length(n_v), maxIter);
admm_loss = zeros(length(n_v), maxIter);
cease_loss = zeros(length(n_v), maxIter);

for idx = 1:length(n_v)
sigmas = [1.0, 1.0, 1.0, 1.0, 1.0];
simOpts = struct('coh', coh, 'a', 1.5, 'alpha', ones(1, r), 'sigmas', sigmas);
simData = genData2(l, n_v(idx), r, C, simOpts);
% Vanilla -ista
istaOpts = struct('lam', lam, 'mode', 'ISTA');
[~, ~, ista_out] = vanillaAls(simData.Xcs, simData.Hcs, maxIter,istaOpts);
% Vanilla -fista
fistaOpts = struct('lam', lam, 'mode', 'FISTA');
[~, ~, fista_out] = vanillaAls(simData.Xcs, simData.Hcs, maxIter, fistaOpts);
% ADMM
rho = 50;
admmOpts = struct('noise_aware', 0, 'rho', rho, 'lam', lam);
[Wcs, Hcs, W, out] = admmAls(simData.Xcs, simData.Hcs, maxIter, admmOpts);
% cease
ceaseOpts = struct('a', 0.15 * m / n, 'noise_aware', 0, 'lam', lam);
[Wcs_, Hcs_, W_, out_] = ceaseAls(simData.Xcs, simData.Hcs, maxIter, ceaseOpts);
% save to cease
ista_loss(idx, :) = ista_out.loss;
fista_loss(idx, :) = fista_out.loss;
admm_loss(idx, :) = out.loss;
cease_loss(idx, :) = out_.loss;
end

set_fig('units','inches','width', 8,'height', 2,'font','Times New Roman','fontsize', 10);
for idx = 1:3
subplot(1, 3, idx)
hold on;
% plot(log(ista_loss(idx, :)))
plot(log(fista_loss(idx, :)))
plot(log(admm_loss(idx, :)))
plot(log(cease_loss(idx, :)))
xlim([0, 30])
box on;
legend('FISTA', 'ADMM', 'CEASE')
title(sprintf('Nonsmooth, n=%d', n_v(idx)))  
hold off
end
save('../../output/synthetic_B.mat', 'ista_loss', 'fista_loss', 'admm_loss','cease_loss','maxIter')


