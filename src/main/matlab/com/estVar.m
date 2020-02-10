function [e, e_] = estVar(rho, Hcs, sigmas)
% Compute the theoritcal largest eigenvalue of the covariance of the
% estimation of W
C = numel(Hcs);
sigmas2 = sigmas.^2;
[r, ~] = size(Hcs{1}) ; 

Z = 0;
Z_ = 0;
ss2 = sum( 1./ sigmas2)^2;
for c = 1:C
    HcHc =  Hcs{c} * Hcs{c}';
    Lc = inv(eye(r) + 1 / rho * HcHc);
    Z = Z + Lc * HcHc * Lc * sigmas2(c);
    Z_ = Z_ + Lc * HcHc * Lc * (1 / sigmas2(c) / ss2);
end
% eigenval omitting noise
e = diag(Z / (C *rho)^2);
% eigenval consdering noise 
e_ = diag(Z_ / rho^2);
end

