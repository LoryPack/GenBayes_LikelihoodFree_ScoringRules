% Plot the number of people the queue for a single scenario

function h = plot_queue_length(y, w)

    [arr_dep_times, queue_delta] = num_people(y, w);
    num_events = length(arr_dep_times);

    h = figure;

    for i = 1 : (num_events - 1)
        
        cur_queue_size = sum(queue_delta(1:i));
        
        line_beg_cur = arr_dep_times(i);
        line_end_cur = arr_dep_times(i+1);
        
        line([line_beg_cur, line_end_cur], [cur_queue_size, cur_queue_size], 'LineWidth', 16);
        set(gca,'FontSize', 22);
        
        disp([line_beg_cur, line_end_cur]);

    end
    
    xlim([0, arr_dep_times(num_events)]);
    xlabel('Time');
    ylabel('Queue Length');

end