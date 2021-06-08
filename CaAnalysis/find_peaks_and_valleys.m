function [peaks, peak_locs, valleys, valley_locs, num_peaks] = ...
                                find_peaks_and_valleys(I, min_peak_prominence, min_peak_width)
                            
    % Finds peaks and valleys in time traces.

    num_cells = size(I,1);
    num_peaks = zeros(1,num_cells);
    peaks = cell(1,num_cells);
    valleys = cell(1,num_cells);
    peak_locs = cell(1,num_cells);
    valley_locs = cell(1,num_cells);
    for i=1:num_cells
        I_i = I(i,:);
        [peaks{i}, peak_locs{i}] = findpeaks(I_i, 'MinPeakProminence', min_peak_prominence, 'MinPeakWidth', min_peak_width);
        [valleys{i}, valley_locs{i}] = findpeaks(-I_i, 'MinPeakProminence', min_peak_prominence, 'MinPeakWidth', min_peak_width);
        num_peaks(i) = length(peaks{i});
    end
end