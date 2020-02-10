function y = softhreolding(x, lam)
y = sign(x) .* max(abs(x) - lam, 0);
end

