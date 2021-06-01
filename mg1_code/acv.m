% Esimate autocovariances with mean x_bar

function acv_vec = acv(x, x_bar, n_lags)

    n = length(x);

    acv_vec = zeros(n_lags, 1);

    for k = 1 : n_lags
        
        acv_vec(k) = sum((x(1:n-k) - x_bar).*(x(k+1:n) - x_bar))/n;
        
    end
    
    acv_vec = [sum((x - x_bar).^2)/n; acv_vec];
    
end