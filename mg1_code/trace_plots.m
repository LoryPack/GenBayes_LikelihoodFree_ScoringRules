% Create trace plots

labels = {'Iteration'};
scenario = 'rare';
filenames = {strcat(scenario,'_0_0_0_16_1'), ...
            strcat(scenario,'_1_1_1_16_1'), ...
            strcat(scenario,'_0_0_0_16_1'), ...
            strcat(scenario,'_1_1_1_16_1'), ...
            strcat(scenario,'_0_0_0_16_1'), ...
            strcat(scenario,'_1_1_1_16_1')};

%time_ratios = [3.9, 1, 3.9, 1, 3.9, 1]; % Frequent
%time_ratios = [1.8, 1, 1.8, 1, 1.8, 1]; % Intermediate
time_ratios = [1.8, 1, 1.8, 1, 1.8, 1]; % Rare

thin = 25;
thin_vec = floor(time_ratios * thin);

x_ticks = [0, 1000, 2000, 3000, 4000];

% Frequent
%x_tick_labels = {{'0', '195,000', '390,000', '585,000', '780,000'}, ...
%                  {'0', '50,000', '100,000', '150,000', '200,000'}, ...
%                  {'0', '195,000', '390,000', '585,000', '780,000'}, ...
%                  {'0', '50,000', '100,000', '150,000', '200,000'}, ...
%                  {'0', '195,000', '390,000', '585,000', '780,000'}, ...
%                  {'0', '50,000', '100,000', '150,000', '200,000'}};

% Intermediate
%x_tick_labels = {{'0', '45,000', '90,000', '135,000', '180,000'}, ...
%                  {'0', '25,000', '50,000', '75,000', '100,000'}, ...
%                  {'0', '45,000', '90,000', '135,000', '180,000'}, ...
%                  {'0', '25,000', '50,000', '75,000', '100,000'}, ...
%                  {'0', '45,000', '90,000', '135,000', '180,000'}, ...
%                  {'0', '25,000', '50,000', '75,000', '100,000'}};

% Rare
x_tick_labels = {{'0', '45,000', '90,000', '135,000', '180,000'}, ...
                  {'0', '25,000', '50,000', '75,000', '100,000'}, ...
                  {'0', '45,000', '90,000', '135,000', '180,000'}, ...
                  {'0', '25,000', '50,000', '75,000', '100,000'}, ...
                  {'0', '45,000', '90,000', '135,000', '180,000'}, ...
                  {'0', '25,000', '50,000', '75,000', '100,000'}};

burn_in_frac = 1/11;

all_upd_lim = 110000;
plot_xlim = (all_upd_lim - all_upd_lim*burn_in_frac)/thin;

%plot_ylim = {[6, 9], [7, 10], [-2.5, -1]}; % Frequent
%plot_ylim = {[3, 4.5], [2, 4], [-2.5, -1]}; % Intermediate
plot_ylim = {[-1, 3], [-2, 12], [-5, -4]}; % Rare

gray = [0.8, 0.8, 0.8];

for k = 1 : length(filenames)
    
    load(filenames{k});
    
    eta = [theta(1), theta(2)-theta(1), log(theta(3))];
    burn_in = numiter*burn_in_frac;
    
    if k == 1 || k == 2
        
        plot_pars = 1;
        legend_cell = {'eta_1'};
        
    elseif k == 3 || k == 4
        
        plot_pars = 2;
        legend_cell = {'eta_2'};
                
    elseif k == 5 || k == 6
        
        plot_pars = 3;
        legend_cell = {'eta_3'};
        
    else
        
        plot_pars = 1:3;
        legend_cell = {'eta_1', 'eta_2', 'eta_3'};
        
    end
    
    h = figure;
    
    plot(par_mat(burn_in+thin_vec(k):thin_vec(k):end, plot_pars), '.', 'Color', gray, 'MarkerSize', 15); 
    set(gca,'FontSize', 16); 
    hold on;
    plot([0, 4000], [eta(ceil(k/2)), eta(ceil(k/2))],'black', 'LineWidth', 2);

    xlabel('Iteration');
    
    set(gca,'XTick', x_ticks);
    set(gca,'XTickLabel', x_tick_labels{k});

    ylim(plot_ylim{ceil(k/2)});
    xlim([0, plot_xlim]);
    
    plotfile = strcat('plot_',scenario,'_',num2str(k),'.eps');
    print(h, '-depsc2', plotfile);
    
end