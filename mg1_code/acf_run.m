% Compute autocorrelation functions

run = strcat(scen,'_', num2str(shift), '_', num2str(range), '_', num2str(rate), '_', num2str(numupd_met));
acf_file = strcat('acf_', run);

par_mat_cell = cell(1, n_inits);
mean_mat = zeros(n_inits, 3);

acv_cell = cell(1, n_inits);
acv_cell_fft = cell(1, n_inits);

for t = 1 : n_inits

    cur_file = strcat(run, '_', num2str(t));
    load(cur_file);
    
    burn_in = numiter/10;

    acv_cell{t} = zeros(n_lags+1, 3);
    acv_cell_fft{t} = zeros(numiter-burn_in, 3);
    
    par_mat_cell{t} = par_mat;
    mean_mat(t, :) = mean(par_mat(burn_in+1:end,:));

end

mean_tot = mean(mean_mat, 1);

for t = 1 : n_inits
    
    for i = 1 : 3
        
        acv_cell{t}(:, i) = acv(par_mat_cell{t}(burn_in+1:end, i), mean_tot(i), n_lags);
        acv_cell_fft{t}(:, i) = autocov(par_mat_cell{t}(burn_in+1:end, i), mean_tot(i));

        disp(i);
        
    end
    
end

acv_mean = zeros(n_lags+1, 3);
acv_mean_fft = zeros(numiter-burn_in, 3);

for t = 1 : n_inits

    acv_mean = acv_mean + acv_cell{t};
    acv_mean_fft = acv_mean_fft + acv_cell_fft{t};

end

acv_mean = acv_mean / n_inits;
acv_mean_fft = acv_mean_fft / n_inits;

acf_mean = bsxfun(@rdivide, acv_mean, acv_mean(1,:));
acf_mean_fft = bsxfun(@rdivide, acv_mean_fft, acv_mean_fft(1,:));

save(acf_file, 'mean_mat', 'acf_mean', 'acf_mean_fft');