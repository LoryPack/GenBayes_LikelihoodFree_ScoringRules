% Create table of autocorrelation time estimates

scen_tag = [0, 0, 0; 1, 0, 0; 0, 1, 0; 0, 0, 1; 1, 1, 1];
scen_names = {'frequent', 'inter', 'rare'};

method_tags = {'Basic', 'Basic + Shift', 'Basic + Range', 'Basic + Rate', 'Basic + All'};

num_par = 3;
num_methods = size(scen_tag, 1);
num_scen = length(scen_names);

tau_cell = cell(num_methods, num_scen);
tau_cell_adj = cell(num_methods, num_scen);

time_mat = [0.13, 0.24, 0.24, 0.27, 0.51; 0.50, 0.61, 0.62, 0.65, 0.89];

for q = 1 : num_scen
    
    if q == 1
        
        numupd_met = 1;
        time_per_one = time_mat(1, :);
        
    else
        
        numupd_met = 16;
        time_per_one = time_mat(2, :);
        
    end
    
    for r = 1 : num_methods
        
        acf_file = strcat('acf_', scen_names{q}, '_', num2str(scen_tag(r, 1)), '_', num2str(scen_tag(r, 2)), '_', num2str(scen_tag(r, 3)), '_', num2str(numupd_met));
        load(acf_file);

        tau_round = sprintf('%1.2g, %1.2g, %1.2g', tau_vec);
        tau_round = str2num(tau_round);
        
        tau_cell{r, q} = tau_round;
        
        tau_adj = time_per_one(r)*tau_round;
        
        tau_adj_round = sprintf('%1.2g, %1.2g, %1.2g', tau_adj);        
        tau_adj_round = str2num(tau_adj_round);
        
        tau_cell_adj{r, q} = tau_adj_round;
      
    end
    
end

tau_mat = cell2mat(tau_cell);
tau_mat_adj = cell2mat(tau_cell_adj);

par_label = '$\\eta_{1}$ & $\\eta_{2}$ & $\\eta_{3}$';
header = strcat('Parameter & ', par_label, ' & ', par_label, ' & ', par_label, ' & ', 'Freq./Inter./Rare', ' & ', par_label, ' & ', par_label, ' & ', par_label, '\\\\ \n');

fid = fopen('tau_table.tex', 'w');
fprintf(fid, '\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|} \n');
fprintf(fid, '\\hline \n');
fprintf(fid, 'Scenario & \\multicolumn{3}{c|}{Frequent} & \\multicolumn{3}{c|}{Intermediate} & \\multicolumn{3}{c|}{Rare} & Time (ms) & \\multicolumn{3}{c|}{Frequent} & \\multicolumn{3}{c|}{Intermediate} & \\multicolumn{3}{c|}{Rare} \\\\ \n');
fprintf(fid, '\\hline \n');
fprintf(fid, header);
fprintf(fid, '\\hline \n');

for r = 1 : num_methods

    table_line = strcat(method_tags{r}, '\t & %8g & %8g & %8g & %8g & %8g & %8g & %8g & %8g & %8g & %s & %8g & %8g & %8g & %8g & %8g & %8g & %8g & %8g & %8g \\\\');                 
    timings = strcat(num2str(time_mat(1, r), '%1.2f'), '/', num2str(time_mat(2, r), '%1.2f'), '/', num2str(time_mat(2, r), '%1.2f'));
    fprintf(fid, table_line, tau_mat(r, :), timings, tau_mat_adj(r, :));
    fprintf(fid, '\n');

end

fprintf(fid, '\\hline \n');
fprintf(fid, '\\end{tabular}\n');
fclose(fid);