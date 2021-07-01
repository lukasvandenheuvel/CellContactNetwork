function [TL,I, background, foreground, meanI] = load_timelapse_and_intensity_levels(path_to_timelapse, segmentation)
    % --------------------------------------------------------------------
    % This function calculates the intensity levels over time of cells on a
    % timelapse video.
    % The intensity levels are extracted from the timelapse.
    % The cell labels are defined by the segmentation.
    %
    % Inputs
    % ------
    %   path_to_timelapse (str)
    %   Path to the timelapse file.
    %
    %   segmentation (MxN array)
    %   On the segmentation image, each cell is labeled with a unique
    %   value.
    %
    % Outputs
    % -------
    %   I (num_cells x num_timepoints array)
    %   Contains the mean intensity levels of cells.
    %
    %   background (1 x num_timepoints vector)
    %   Mean intensity levels of background (area where there are no cells).
    %
    %   foreground (1 x num_timepoints vector)
    %   Mean intensity levels of background (area where there are cells).
    %
    %   meanI (1 x num_timepoints vector)
    %   Mean intensity levels of frames.
    % --------------------------------------------------------------------
    
    % Get number of timepoints in timelapse and number of cells
    info = imfinfo(path_to_timelapse);
    num_timepoints = length(info);
    cell_labels = unique(segmentation);
    cell_labels(cell_labels==0) = []; % Remove 0 (= background)
    num_cells = length(cell_labels);
    
    % Get background and foreground masks
    background_mask = logical(1-(segmentation>0));
    foreground_mask = ~background_mask;
    
    % Init arrays
    I = zeros(num_cells, num_timepoints);   % intensity levels
    background = zeros(1, num_timepoints);  % background intensity level
    foreground = zeros(1, num_timepoints);  % foreground background level
    meanI = zeros(1, num_timepoints);       % mean intensity level of whole frames
    
    % Init timelapse array
    width = info(1).Width;
    height = info(1).Height;
    TL = zeros(num_timepoints, height, width); % timelapse
    
    % Loop over timepoints
    for t=1:num_timepoints
        frame = imread(path_to_timelapse,t);
        stats = regionprops(segmentation,frame,'MeanIntensity');
        intensities = [stats.MeanIntensity]';
        I(:,t) = intensities(cell_labels)';
        background(t) = mean2(frame(background_mask));
        foreground(t) = mean2(frame(foreground_mask));
        meanI(t) = mean2(frame);
        TL(t,:,:) = frame;
    end
end

