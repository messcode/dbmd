function sigmas = calSimgas(Xcs, Wcs, Hcs)
%Calculate the sigmas 

C = length(Xcs);
sigmas = zeros(C, 1);
for c = 1:C
    sigmas(c) = ( sum(sum((Xcs{c} - Wcs{c} * Hcs{c}) .^2)) + 1.0) / (4.0 + numel(Xcs{c}));
end

end

