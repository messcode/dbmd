addpath(genpath('matlab'));
n_v = 100:100:6000; C = 5;
r = 20;
sigmas = [1.0, 1.0, 1.0, 1.0, 1.0];
REP = 50;
%% Dataset A
eigs_min_A = zeros(length(n_v), 1);
eigs_max_A = zeros(length(n_v), 1);
for idx = 1:length(n_v)
    emin = zeros(REP, 1);
    emax = emin;
    for rep = 1:REP
        simOpts = struct('coh', coh, 'a', 1.5, 'p', 1 / (r-1), 'sigmas', sigmas);
        simData = genData(l, n_v(idx), r, C, simOpts);
        HHt = 0;
        for c = 1:C
             Hc = simData.Hcs{c};
             HHt = HHt + Hc * Hc';
        end
        emin(rep) = eigs(HHt, 1, 'SM');
        emax(rep) = eigs(HHt, 1, 'LM');
    end
    eigs_min_A(idx) = mean(emin);
    eigs_max_A(idx) = mean(emax);
end
%% Dataset B
eigs_min_B = zeros(length(n_v), 1);
eigs_max_B = zeros(length(n_v), 1);
for idx = 1:length(n_v)
    emin = zeros(REP, 1);
    emax = emin;
    for rep = 1:REP
        simOpts = struct('coh', coh, 'a', 1.5, 'alpha', ones(1, r), 'sigmas', sigmas);
        simData = genData2(l, n_v(idx), r, C, simOpts);
        HHt = 0;
        for c = 1:C
             Hc = simData.Hcs{c};
             HHt = HHt + Hc * Hc';
        end
        emin(rep) = eigs(HHt, 1, 'SM');
        emax(rep) = eigs(HHt, 1, 'LM');
    end
    eigs_min_B(idx) = mean(emin);
    eigs_max_B(idx) = mean(emax);
end
save('../../output/synthetic/Hc_eigenvalues.mat', 'eigs_min_A', 'eigs_max_A', 'eigs_min_B', 'eigs_max_B', 'n_v');
%% plot

hold on
% plot(eigs_min_A)
plot(log10(n_v), log10(eigs_max_A))
plot(log10(n_v), log10(eigs_max_A ./ eigs_min_A))
% plot(eigs_min_B)
plot(log10(n_v), log10(eigs_max_B))
plot(log10(n_v), log10(eigs_max_B ./ eigs_min_B))
legend('Bmax', 'Bkappa', 'Hmax', 'Hkappa')
hold off;
xx = load('../../output/Hc_eigenvalues.mat');

n = 10;
W1 = rand(n, n);
W2 = rand(n, n);
