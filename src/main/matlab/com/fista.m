function W = fista(Xc, Hc, W, lam, a, gradF)
    HHt = Hc * Hc';
    gradF = (W * HHt - Xc * Hc') - gradF;
    L = eigs(HHt, 1);
    maxIter = 100;
    normW = norm(W);
    tol = 1e-6; 
    preWc = W;
    Y = W;
    t = 1;
    for nIter = 1:maxIter
        grad = Y * HHt - Xc * Hc' + a * (W - Y) + gradF;
        W = Y - grad / L;
        W = softhreolding(W, lam / L);
        numerator = t - 1;
        t = .5 + sqrt(1 + 4 * t * t);
        Y = W + (numerator / t) * (W - preWc);
        preWc = W;
        if norm(preWc - W) / normW < tol && nIter > 5
            break
        end
    end
end
