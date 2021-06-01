% Create table of parameter estimates

scen_name = {'rare'};
scen_tag = [0, 0, 0; 1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 1];
numupd_met = 16;

method_tags = {'Basic', 'Basic + Shift', 'Basic + Range', 'Basic + Rate', 'Basic + All'};

num_par = 3;
num_methods = size(scen_tag, 1);
n_inits = 5;

mean_cell = cell(num_methods, 1);

for r = 1 : num_methods

    mean_tmp_mat = zeros(n_inits, num_par);

    for p = 1 : n_inits

        run_file = strcat(scen_name{1}, '_', num2str(scen_tag(r, 1)), '_', num2str(scen_tag(r, 2)), '_', num2str(scen_tag(r, 3)), '_', num2str(numupd_met), '_', num2str(p));
        load(run_file);
        disp(numiter);
        
        burn_in = numiter/10;
        mean_tmp_mat(p, :) = mean(par_mat(burn_in+1:end, :));
        clear par_mat;

    end
    
    a = mean_tmp_mat';
    ste_mean_est = std(a, 0, 2)/sqrt(n_inits);
    
    mean_cell{r} = reshape([mean(a, 2), mean(a, 2) - 2*ste_mean_est, mean(a, 2) + 2*ste_mean_est, ste_mean_est]', 12, 1)';

end

acc_rates = [acc_met, acc_shift, acc_range_scale, acc_rate_scale] / numiter;

mean_mat = cell2mat(mean_cell);

par_label = '\\multicolumn{3}{c|}{$\\eta_{1}$} & \\multicolumn{3}{c|}{$\\eta_{2}$} & \\multicolumn{3}{c|}{$\\eta_{3}$}';
header_1 = strcat('Parameter & ', par_label, '\\\\ \n');
est_label = ' Mean & CI & std. err. ';
header_2 = strcat('Estimates &', est_label, ' & ', est_label, ' & ', est_label, '\\\\ \n');

fid = fopen('mean_table.tex', 'w');
fprintf(fid, '\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|} \n');
fprintf(fid, '\\hline \n');
fprintf(fid, header_1);
fprintf(fid, '\\hline \n');
fprintf(fid, header_2);
fprintf(fid, '\\hline \n');

for r = 1 : num_methods

    table_line = strcat(method_tags{r}, '\t & %4.4f & (%4.4f, %4.4f) & %4.5f & %4.4f & (%4.4f, %4.4f) & %4.5f & %4.4f & (%4.4f, %4.4f) & %4.5f \\\\');                 
    fprintf(fid, table_line, mean_mat(r, :));
    fprintf(fid, '\n');

end

fprintf(fid, '\\hline \n');
fprintf(fid, '\\end{tabular}\n');
fclose(fid);