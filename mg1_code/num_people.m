% Compute the number of people currently in the queue

function [arr_dep_times, queue_delta] = num_people(y, w)

    n = length(y);

    v_tag = [cumsum(w); ones(1, n)]';
    x_tag = [cumsum(y); -ones(1, n)]';

    arr_dep = zeros(2*n+1, 2);

    v_offset = 1;
    x_offset = 1;
    arr_dep_offset = 2;

    while v_offset <= n

        if v_tag(v_offset, 1) < x_tag(x_offset, 1)

            arr_dep(arr_dep_offset, :) = v_tag(v_offset, :);
            v_offset = v_offset + 1;

        else

            arr_dep(arr_dep_offset, :) = x_tag(x_offset, :);
            x_offset = x_offset + 1;

        end

        arr_dep_offset = arr_dep_offset + 1;

    end

    arr_dep(arr_dep_offset:end, :) = x_tag(x_offset:end,:);

    arr_dep_times = arr_dep(1:arr_dep_offset-1, 1);
    queue_delta = arr_dep(1:arr_dep_offset-1, 2);

end