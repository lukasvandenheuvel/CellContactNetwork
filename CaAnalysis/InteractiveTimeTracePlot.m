% 19/05/2021 by Lukas van den Heuvel
% This script will open an interactive plot which asks the user to click on
% a cell. It then shows the timetrace of that cell.
clear all
close all
clc

intial_path = 'M:\tnw\bn\dm\Shared';

% Choose data
[timelapse_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the timelapse file', intial_path);
timelapse_path = fullfile(directory, timelapse_file);
segmented_file = uigetfile({'*.tif';'*.png'}, 'Choose the tile segmented image', directory);
segmented_path = fullfile(directory, segmented_file);
network_file = uigetfile({'*.mat'}, 'Choose the tile network measurements file', directory);
network_path = fullfile(directory, network_file);

%% Choose frame rate
sampling_rate = input('Enter frame rate in Hz: ');
min_peak_prominence = input('Enter a peak prominence: ');

% Load segmentation file
segmentation = imread(segmented_path);
segmented_rgb = label2rgb(segmentation,'jet',[0,0,0],'shuffle');
num_cells = max(segmentation(:));

% Load intensity levels
disp('Loading intensity levels...')
[TL, I, background, foreground, meanI] = load_timelapse_and_intensity_levels(timelapse_path, segmentation);
time = get_time_axis(I, sampling_rate);

info = imfinfo(timelapse_path);
num_timepoints = length(info);
width = info(1).Width;
height = info(1).Height;

%% Load network data
data = load(network_path);
x_nodes = data.centroid1;
y_nodes = data.centroid0;

max_frame_nr = num_timepoints;
%% Show segmented image
figure(1)
imshow(segmented_rgb)

%% Normalize time traces with baseline
normI = (I - background) ./ background;

%% Peak finder
min_peak_width = 0;
[peaks, peak_locs, valleys, valley_locs, num_peaks] = ...
                                find_peaks_and_valleys(normI, min_peak_prominence, min_peak_width);

figure(2)
histogram(num_peaks)
xlabel('Number of peaks')
ylabel('Frequency')
unique_spike_nrs = unique(num_peaks);
unique_spike_nrs(unique_spike_nrs==0) = [];

%% Find spiking cells and normalize their intensity
spiking_cells = num_peaks > 3;
I_spiking = normI(spiking_cells, 1:max_frame_nr);
I_spiking_norm = zeros(size(I_spiking));
for i = 1:sum(spiking_cells)
    I_i = I_spiking(i,:);
    norm_I_i = (I_i - min(I_i)) / (max(I_i) - min(I_i));
    I_spiking_norm(i,:) = norm_I_i;
end

%% Interactive plot of time traces
% This plot will first show the timelapse image. Then it asks the user to
% click on cells, and it will show the corresponding time traces
figure(3)
set(gcf,'PaperOrientation','landscape');
set(gcf,'Color','w','Units','inches','Position',[1 1 18 9])
colors = jet(length(unique_spike_nrs));

% Show time frames
subplot(2,4,[1 2 5 6])
for t=1:max_frame_nr
    frame = reshape(TL(t,:,:), [height, width,1]);
    frame = (frame - min(frame(:))) / (max(frame(:)) - min(frame(:))); % Normalize
    imshow(frame)
    hold on
    title(['t = ', num2str(time(t),3), ' s.'])
    drawnow
end

% Indicate the cells that spiked
figure(3)
for i = 1:length(unique_spike_nrs)
    cell_nrs = find(num_peaks == unique_spike_nrs(i));
    x_centroid = data.centroid1(cell_nrs);
    y_centroid = data.centroid0(cell_nrs);
    plot(x_centroid, y_centroid, '*', 'Color', colors(i,:))
end

title('Click on a cell to see its time trace. Right-click to stop')
button = 1;

% choose first cell
cell_nr1 = 0;
while cell_nr1 == 0
    disp('choose cell')
    [x,y] = ginput(1);
    cell_nr1 = segmentation(round(y),round(x));
end
[boundary1, ymax1, ymin1] = get_cell_specifics(cell_nr1, normI, segmentation, max_frame_nr);
I1 = normI(cell_nr1,:);
locs = peak_locs{cell_nr1};

% plot boundary
figure(3)
subplot(2,4,[1 2 5 6])
plot(boundary1(:,2), boundary1(:,1), '-g');
% plot time trace
subplot(2,4,3:4)
plot(time, I1, '-g', time(locs), I1(locs), 'xk')
ylim([ymin1-0.1, ymax1+0.1])
title(['Time trace cell ', num2str(cell_nr1)])

cell_nr2 = cell_nr1;

while button == 1 % stop while loop if the user clicked the right mouse (this will return button = 3).
    
    % Choose a new cell by clicking
    cell_nr1 = cell_nr2;
    disp('choose cell')
    [x,y,button] = ginput(1);
    cell_nr2 = segmentation(round(y),round(x));
    % if there is no cell at the location, skip current while-loop iteration
    if cell_nr2 == 0
        cell_nr2 = cell_nr1;
        continue
    end
    
    % Plot the boundary of the 'old' cell red.
    figure(3)
    subplot(2,4,[1 2 5 6])
    plot(boundary1(:,2), boundary1(:,1), '-r');
    
    % Get boundaries and peak locations of the current 2 cells
    [boundary1, ymax1, ymin1] = get_cell_specifics(cell_nr1, normI, segmentation, max_frame_nr);
    I1 = normI(cell_nr1,:);
    plocs1 = peak_locs{cell_nr1};
    vlocs1 = valley_locs{cell_nr1};
    [boundary2, ymax2, ymin2] = get_cell_specifics(cell_nr2, normI, segmentation, max_frame_nr);
    I2 = normI(cell_nr2,:);
    plocs2 = peak_locs{cell_nr2};
    vlocs2 = valley_locs{cell_nr2};

    % plot boundaries of current two cells as magentha and red
    figure(3)
    subplot(2,4,[1 2 5 6])
    plot(boundary1(:,2), boundary1(:,1), '-m');
    plot(boundary2(:,2), boundary2(:,1), '-g');
    
    % plot time traces of the current 2 cells in 2 subplots
    figure(3)
    subplot(2,4,3:4)
    plot(time, I1, '-m', time(plocs1), I1(plocs1), 'xk', time(vlocs1), I1(vlocs1), 'xk')
    ylim([ymin1-0.1, ymax1+0.1])
    title(['Time trace cell ', num2str(cell_nr1)])
    xlabel('time (s)')
    ylabel('Fluo-8 (relative to mean)')
    
    figure(3)
    subplot(2,4,7:8)
    plot(time, I2, '-g', time(plocs2), I2(plocs2), 'xk', time(vlocs2), I2(vlocs2), 'xk')
    ylim([ymin2-0.1, ymax2+0.1])
    title(['Time trace cell ', num2str(cell_nr2)])
    xlabel('time (s)')
    ylabel('Fluo-8 (relative to mean)')
end

%%
max_time_point = 90;
max_time = round(sampling_rate * max_time_point);
I1 = normI(870,1:max_time);
I2 = normI(893,1:max_time);

green = [124,252,0] / 255;

figure()
plot(time(1:max_time),I1,'c', 'LineWidth', 2)
hold on
plot(time(1:max_time),I2, 'Color', green, 'LineWidth', 2)
ylim([0,1.3])

xlabel('Time (s)')
ylabel('Ca^{2+} level (a.u.)')
legend('Cell 1', 'Cell 2')
set(gca, 'LineWidth', 1,'FontSize',12)
set(gcf,'Color','w','Units','inches','Position',[9 1 10 2])
saveas(gcf,fullfile(directory, 'CaImaging.png'))

%% Functions
function [boundary, ymax, ymin] = get_cell_specifics(cell_nr, normI, segmented, max_frame_nr)

    [B,~] = bwboundaries(segmented==cell_nr,'noholes');
    boundary = B{1};

    ymax = max(normI(cell_nr,1:max_frame_nr));
    ymin = min(normI(cell_nr,1:max_frame_nr));
    
end
%%
