% Compute autocorrelations for a particular scenario

clear all;

num_methods = 5;
scen = 'rare';

scen_tag = [0, 0, 0; 1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 1];

numupd_met = 16;

n_inits = 5;
n_lags = 20;
thin = 1;

for r = 1 : num_methods

    shift = scen_tag(r, 1);
    range = scen_tag(r, 2);
    rate = scen_tag(r, 3);
    
    acf_run;

end