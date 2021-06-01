% Logarithm of the normal density

function lprob = lpdf(x, mu, sigma)

    lprob = -0.5 * (log(2 * pi * sigma.^2) + ((x - mu) ./ sigma).^2);

end