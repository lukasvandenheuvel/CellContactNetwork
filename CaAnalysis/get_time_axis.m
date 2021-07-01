function time_axis = get_time_axis(I, sampling_frequency)
    num_timepoints = size(I,2);
    total_time = num_timepoints / sampling_frequency;
    time_axis = linspace(0, total_time, num_timepoints);
end