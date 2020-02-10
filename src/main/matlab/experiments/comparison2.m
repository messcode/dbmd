addpath(genpath('matlab'));
%% change n smooth
l = 20; n = 100; r = 20; C = 5; coh = 4;
r_ = r - 1;
m = l * r_ - (r_ - 1) * coh;
n_v = [100, 500, 1000];
maxIter = 100;
% save loss to matrix
admm_loss = zeros(length(n_v), maxIter);
cease_loss = zeros(length(n_v), maxIter);
set_fig('units','inches','width', 8,'height', 2,'font','Times New Roman','fontsize', 10);
for idx = 1:length(n_v)
sigmas = [1.0, 1.0, 1.0, 1.0, 1.0];
simOpts = struct('coh', coh, 'a', 1.5, 'alpha', ones(1, r), 'sigmas', sigmas);
simData = genData2(l, n_v(idx), r, C, simOpts);
% ADMM
rho = 100;
admmOpts = struct('noise_aware', 0, 'rho', rho);
[Wcs, Hcs, W, out] = admmAls(simData.Xcs, simData.Hcs, maxIter, admmOpts);
% cease
ceaseOpts = struct('a', 0.15 * m / n, 'noise_aware', 0);
[Wcs_, Hcs_, W_, out_] = ceaseAls(simData.Xcs, simData.Hcs, maxIter, ceaseOpts);
% save to cease
admm_loss(idx, :) = out.loss;
cease_loss(idx, :) = out_.loss;
subplot(1, 3, idx)
hold on;
plot(log(out.loss))
plot(log(out_.loss))
xlim([0, maxIter])
box on;
legend('ADMM', 'CEASE')
title(sprintf('Smooth, n=%d', n_v(idx)))  
hold off
end
save('../../output/comparison_smooth2.mat', 'admm_loss','cease_loss','maxIter')
export_fig '../../doc/figs/comparison_vary_n2.png' -r600



%% Noise aware
l = 5; n = 100; r = 4; C = 5; coh = 1;
r_ = r - 1;
m = l * r_ - (r_ - 1) * coh;
sigma_last_v = 1.0 :.5:10; 
emp_ratio_cease = zeros(size(sigma_last_v));
the_ratio = zeros(size(sigma_last_v));
emp_ratio_admm = zeros(size(sigma_last_v));
for sigma_idx = 1:length(sigma_last_v)
rho = 100; sigmas = [1.0, 1.0, 1.0, 1.0, sigma_last_v(sigma_idx)];
REP = 100;
% save the estimation of W
vecW = zeros(m * r, REP);
vecW_ = zeros(m * r, REP);
vecW_admm = zeros(m * r, REP);
vecW_admm_ = zeros(m * r, REP);
% Simlu
simOpts = struct('coh', coh, 'a', 1.5, 'alpha', ones(1, r), 'sigmas', sigmas);
simData = genData2(l, n, r, C, simOpts);
simOpts.Hcs = simData.Hcs;
for idx = 1:REP
simData = genData(l, n, r, C, simOpts);
maxIter = 100;
% CEASE
ceaseOpts = struct('a', 0.15 * m / n, 'noise_aware', 0);
[Wcs, Hcs, W, out] = ceaseAls(simData.Xcs, simData.Hcs, maxIter, ceaseOpts);
ceaseOpts = struct('a', 0.15 * m / n, 'noise_aware', 1);
[Wcs_, Hcs_, W_, out_] = ceaseAls(simData.Xcs, simData.Hcs, maxIter, ceaseOpts);
% ADMM
admmOpts = struct('noise_aware', 0, 'rho', rho);
[Wcs_admm, Hcs_admm, W_admm, out_admm] = admmAls(simData.Xcs, simData.Hcs, maxIter, admmOpts);
admmOpts = struct('noise_aware', 1, 'rho', rho);
[Wcs_admm_, Hcs_admm_, W_admm_, out_admm_] = admmAls(simData.Xcs, simData.Hcs, maxIter, admmOpts);
% save results
vecW(:, idx) = W(:);
vecW_(:, idx) = W_(:);
vecW_admm(:, idx) = W_admm(:);
vecW_admm_(:, idx) = W_admm_(:);
end
emp_e_cease = var(vecW, 0, 2); % 0.0047
emp_e_cease_ = var(vecW_, 0, 2); % 0.0012 0.0011
emp_e_admm = var(vecW_admm, 0, 2); % 0.0047
emp_e_admm_ = var(vecW_admm_, 0, 2); % 0.0012 0.0011
[e, e_] = estVar(rho, simData.Hcs, sigmas);
emp_ratio_cease(sigma_idx) = mean(emp_e_cease_) / mean(emp_e_cease); 
emp_ratio_admm(sigma_idx) = mean(emp_e_admm_) / mean(emp_e_admm); 
the_ratio(sigma_idx) =  mean(e_) / mean(e);
fprintf('Empirical Ratio:   %.4f\n', mean(emp_e_cease_) / mean(emp_e_cease))
fprintf('Theoritical Ratio: %.4f\n', mean(e_) / mean(e));
fprintf('The / Emp = %.4f\n', mean(emp_e_cease) / mean(e_))
end

save('../../output/variance_ratio2.mat', 'emp_ratio_cease', 'emp_ratio_admm', 'the_ratio', 'sigma_last_v');

% figure
set_fig('units','inches','width', 6,'height', 2.25,'font','Times New Roman','fontsize', 10);
subplot(1, 2, 1)
hold on;
plot(sigma_last_v, emp_ratio_cease)
plot(sigma_last_v, the_ratio, '--')
ylabel('Variance ratio')
xlabel('Noise level')
ylim([0, 1])
xlim([1, 10.5])
legend('Empirical', 'Theoritical')
title('CEASE')
box on
hold off

subplot(1, 2, 2)
hold on;
plot(sigma_last_v, emp_ratio_admm)
plot(sigma_last_v, the_ratio, '--')
ylabel('Variance ratio')
xlabel('Noise level')
legend('Empirical', 'Theoritical')
title('ADMM')
ylim([0, 1])
xlim([1, 10.5])
box on
hold off
export_fig '../../doc/figs/comparison_variance_ratio.png' -r600