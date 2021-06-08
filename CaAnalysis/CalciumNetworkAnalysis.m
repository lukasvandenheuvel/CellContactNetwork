clear all
close all

intial_path = 'M:\tnw\bn\dm\Shared';

% Choose data
[timelapse_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the timelapse file', intial_path);
timelapse_path = fullfile(directory, timelapse_file);

segmented_file = uigetfile({'*.tif';'*.png'}, 'Choose the segmented image', directory);
segmented_path = fullfile(directory, segmented_file);
segmentation = imread(segmented_path);
background_mask = logical(1-(segmentation>0));
foreground_mask = ~background_mask;

rgb_file = uigetfile({'*.tif';'*.png'}, 'Choose the RGB image.', directory);
rgb_path = fullfile(directory, rgb_file);
cell_image = imread(rgb_path);
network = uigetfile('.mat','Choose network file',directory); 

%% Load network data and measurements
cd(directory)
network = load(network);

G = graph(network.contact_matrix);
A = full(adjacency(G));
D = diag(sum(A,1));
object_com = [network.centroid0' network.centroid1'];
num_cells = size(A,1);

%% Load stack
info = imfinfo(timelapse_path);
num_timepoints = length(info);
width = info(1).Width;
height = info(1).Height;

% Init arrays
I = zeros(num_cells, num_timepoints);
deltaI = zeros(num_cells, num_timepoints);
normI = zeros(num_cells, num_timepoints);
background = zeros(1, num_timepoints);
foreground = zeros(1, num_timepoints);
meanI = zeros(1, num_timepoints); % mean intensity of slice

% Loop over timepoints
disp('Loading timelapse...')
for t=1:num_timepoints
    calcium_image = imread(timelapse_path,t);
    stats = regionprops(segmentation,calcium_image,'MeanIntensity');
    I(:,t) = [stats.MeanIntensity]';
    background(t) = mean2(calcium_image(background_mask));
    foreground(t) = mean2(calcium_image(foreground_mask));
    meanI(t) = mean2(calcium_image);
    deltaI(:,t) = (I(:,t)-background(t)) ./ (I(:,1)-background(1));
    normI = (I(:,t)-mean(I(:,t)))./mean(I(:,t));
end

%%
figure()
plot(background)
hold on
plot(foreground)
plot(meanI)
legend('Background', 'Foreground', 'meanI')

%% Find peaks and valleys
min_peak_prominence = 0.1;
[peaks, peak_locs, valleys, valley_locs, num_peaks] = ...
                                find_peaks_and_valleys(deltaI, min_peak_prominence);


unique_spike_nrs = unique(num_peaks);
unique_spike_nrs(unique_spike_nrs==0) = [];
spiking_cells = find(num_peaks > 5);
GraphD = distances(G,spiking_cells,spiking_cells);

figure()
imshow(cell_image)
hold on
h = plot(G,'XData',object_com(:,2),'YData',object_com(:,1),'MarkerSize',4,'LineWidth',1,'NodeColor','w','EdgeColor','w')
highlight(h,spiking_cells,'NodeColor',[0 1 0])
labelnode(h,spiking_cells,string(spiking_cells))
h.NodeLabelColor = [0 1 0];
h.NodeFontSize = 25;
set(gcf,'Color','k')
framerate = 0.5; %frames per second
timeaxis = (0:num_timepoints-1)/(framerate*60); %Time in minutes

                
figure()
subplot(1,4,1)
histogram(num_peaks)
xlabel('Number of peaks')
ylabel('Frequency')
set(gca, 'FontSize', 12)

subplot(1,4,2)
plot(timeaxis,deltaI(spiking_cells,:))
xlabel('Time (min)')
ylabel('Normalized fluorescence (a.u.)')
set(gca, 'FontSize', 12)
% leg = legend(string(spiking_cells));
% leg.ItemTokenSize = [10 10];

subplot(1,4,3)
tree = linkage(squareform(GraphD,'tovector'),'single');
[H,T,outperm] = dendrogram(tree,0,'Orientation','left')%,'Labels',string(spiking_cells));
set(H,'LineWidth',1,'Color','k');
xlabel('Topological distance')
set(gca, 'FontSize', 12)
%xlim([0 tree(max(find(tree(:,3)<1)),3)])

subplot(1,4,4)
imagesc('XData', timeaxis, 'CData', I(spiking_cells(outperm),:))
yticklabels ''
ylim([0 length(spiking_cells)+1])
xlabel('Time (min)')
colormap parula
c = colorbar;
c.Label.String = 'Ca^{2+} level';
set(gca, 'FontSize', 12)
set(gcf,'Color','w','Units','inches','Position',[0 0 12 8])

%%
function [peaks, peak_locs, valleys, valley_locs, num_peaks] = ...
                                find_peaks_and_valleys(I, min_peak_prominence)
                            
    % Finds peaks and valleys in time traces.

    num_cells = size(I,1);
    num_peaks = zeros(1,num_cells);
    peaks = cell(1,num_cells);
    valleys = cell(1,num_cells);
    peak_locs = cell(1,num_cells);
    valley_locs = cell(1,num_cells);
    for i=1:num_cells
        I_i = I(i,:);
        [peaks{i}, peak_locs{i}] = findpeaks(I_i, 'MinPeakProminence', min_peak_prominence);
        [valleys{i}, valley_locs{i}] = findpeaks(-I_i, 'MinPeakProminence', min_peak_prominence);
        num_peaks(i) = length(peaks{i});
    end
end