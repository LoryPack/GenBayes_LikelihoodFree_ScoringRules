% Plot the number of people in the queue across time for all scenarios

clear all;

files = {'queue_frequent', 'queue_inter', 'queue_rare'};
files = {'data/my_observation_1'};
y_ticks = {[0, 5, 10, 15, 20, 25], [0, 2, 4, 6, 8], [0, 1]};

for i = 1 : length(files)
    
    load(files{i});
    h = plot_queue_length(y, w);   
    filename = strcat(files{i}, '_plot');
    set(gca,'YTick', y_ticks{i});
    print(h, '-deps2', filename);
    
end