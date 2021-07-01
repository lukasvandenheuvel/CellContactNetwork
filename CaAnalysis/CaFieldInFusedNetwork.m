clear all
close all

intial_path = 'M:\tnw\bn\dm\Shared';

%% Choose data with dialogs

% Fused segmentation
[segmented_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the FUSED segmented image', intial_path);
segmented_path = fullfile(directory, segmented_file);
% Fused RGB
[rgb_fused_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the FUSED RGB image', directory);
rgb_fused_path = fullfile(directory, rgb_fused_file);
% Fused network
[network_file, directory] = uigetfile('.mat','Choose FUSED network file',directory); 
network_path = fullfile(directory, network_file);
% Alignment file (tile in fused)
[alignment_file, directory] = uigetfile('.txt','Choose alignment file',directory); 
alignment_path = fullfile(directory, alignment_file);
% Timelapse
[timelapse_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the timelapse file', directory);
timelapse_path = fullfile(directory, timelapse_file);
% Tile RGB image
[rgb_tile_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the TILE RGB image', directory);
rgb_tile_path = fullfile(directory, rgb_tile_file);

%% Choose frame rate
sampling_rate = input('Enter frame rate in Hz: ');

%% Load images
disp('Loading images...')
segmentation = imread(segmented_path);
rgb_tile = imread(rgb_tile_path);
rgb_fused = imread(rgb_fused_path);
[N,M] = size(rgb_fused);

%% Load network and cell measurements
disp('Loading network...')
network = load(network_path);
G = graph(network.contact_matrix);
x_nodes = network.centroid1;            % x-coordinates of centers of mass 
y_nodes = network.centroid0;            % y-centers of mass measured from the origin
pix_to_um = 6300/length(segmentation); 	% number of micrometers per pixel
area = network.area * pix_to_um^2;      % area in um^2
num_nodes = numnodes(G);

[xt,yt,wt,ht] = get_tile_position_in_fused(alignment_path);

tile_segmentation = segmentation(yt:yt+ht-1, xt:xt+wt-1);
rgb = label2rgb(tile_segmentation,'jet',[0,0,0],'shuffle'); % randomly relabel to RGB colormap for visualization

%% Subgraph of nodes in field
nodes_in_tile = unique(tile_segmentation);
nodes_in_tile(nodes_in_tile==0) = []; % remove 0 (= background)
Gt = subgraph(G, nodes_in_tile);
num_nodes_t = numnodes(Gt);
x_nodes_t = x_nodes(nodes_in_tile) - xt;
y_nodes_t = y_nodes(nodes_in_tile) - yt;

%% Load timelapse and intensity levels per cell
[TL, I, background, foreground, meanI] = load_timelapse_and_intensity_levels(timelapse_path, tile_segmentation);
time = get_time_axis(I, sampling_rate);

% Normalize time traces with baseline
normI = (I - background) ./ background;

% Peak finder (Ca2+ peaks)
min_peak_prominence = 0.1;
min_peak_width = 0;
[peaks, peak_locs, valleys, valley_locs, num_peaks] = ...
                                find_peaks_and_valleys(normI, min_peak_prominence, min_peak_width);

%% Make a heatmap

% find spiking cells and normalize their intensity
max_frame_nr = size(I,2);
spiking_cells = (num_peaks > 3);
I_spiking = normI(spiking_cells, 1:max_frame_nr);
I_spiking_norm = zeros(size(I_spiking));
for i = 1:sum(spiking_cells)
    I_i = I_spiking(i,:);
    norm_I_i = (I_i - min(I_i)) / (max(I_i) - min(I_i));
    I_spiking_norm(i,:) = norm_I_i;
end

% Order spiking cells in dendrogram, based on Graph distance :

% Make distance matrix for spiking cells
D = distances(Gt);
D_spiking = D(spiking_cells, spiking_cells);
% Convert distance matrix to a vector with distances between pairs of nodes:
D_vector = squareform(D_spiking,'tovector');
% Hierarchical clustering based on graph distance:
tree = linkage(D_vector,'single');

%% Graph-theoretical analysis

% Betweenness centrality
bc = 2*centrality(G,'betweenness')/((num_nodes-1)*(num_nodes-2));
bc_t = bc(nodes_in_tile);

%% Plotting                             

figure()

% Fused image
subplot(1,5,1)
imshow(rgb_fused)
hold on
rectangle('Position',[xt, yt, wt, ht], ...
          'EdgeColor',[0,1,0], 'LineWidth',2)
      
% Tile with network
color = ones(num_nodes_t,3);
color(spiking_cells,:) = repmat([0,1,0], [sum(spiking_cells),1]); % give spiking cells a green color
subplot(1,5,2)
imshow(rgb_tile)
hold on
plot(Gt,'XData',x_nodes_t,'YData',y_nodes_t,'MarkerSize',1,'LineWidth',1,'NodeColor',color,'EdgeColor','w')
hold off

% Plot dendogram
subplot(1,5,3)
[H,~,outperm] = dendrogram(tree,0,'Orientation','left');
set(H,'LineWidth',1,'Color','k');
set(gca, 'FontSize', 9, 'LineWidth', 1);
set(gcf,'Color','w','Units','inches','Position',[9 1 1.5 3.5]);
yticks([])
xlabel('Graph distance')

% Plot heatmap
subplot(1,5,4)
imagesc('XData', time(1:max_frame_nr), 'YData', 1:sum(spiking_cells), 'CData', I_spiking_norm(outperm,1:max_frame_nr))
xlabel('Time (s)')
xlim([1,time(max_frame_nr)])
ylabel('spiking cell #')
ylim([1,sum(spiking_cells)])

colormap parula
c = colorbar;
c.Label.String = 'Ca^{2+} level (normalized \DeltaF/F_{mean})';
set(gca, 'FontSize', 9, 'LineWidth', 1)
set(gcf,'Color','w','Units','inches','Position',[9 1 3 3.5])
saveas(gcf,fullfile(directory, 'Heatmap.png'))

% Number of peaks versus betweenness centrality
subplot(1,5,5)
semilogx(bc_t, num_peaks, 'b.')
xlabel('Betweenness')
ylabel('Number of Ca2+ peaks')

set(gcf,'Color','w','Units','inches','Position',[1 1 15 5])

%% Functions
function [x,y,w,h] = get_tile_position_in_fused(alignment_path)
    % This function returns the position (x,y - coordinates, width and height)
    % of a tile in the fused image.
    % The coordinates are stored in alignment file, which is saved by Fiji
    % when you run the macro AlignTileInFused.ijm.
    %
    % INPUT
    % ------
    % alignment_path (string)
    % The path to the .txt file which contains the coordinates.
    %
    % RETURNS
    % -------
    % (x,y) (integers)
    % The x,y coordinate of the upper-left corner of the tile inside the
    % fused image in pixels.
    %
    % (w,h) (integers)
    % Width and height of the tile in pixels.
    
    % Open text files
    alignment_txt = fopen(alignment_path);

    % Read all lines & collect in cell array
    alignment_results = textscan(alignment_txt,'%s','delimiter','\n');

    x = inf;
    y = inf;
    w = inf;
    h = inf;
    for i=1:length(alignment_results{1})
        line = alignment_results{1}{i};
        if contains(line, 'TileXposition')
            split_line = split(line, ' = ');
            x = str2double(split_line{2});
        elseif contains(line, 'TileYposition')
            split_line = split(line, ' = ');
            y = str2double(split_line{2});
        elseif contains(line, 'TileWidth')
            split_line = split(line, ' = ');
            w = str2double(split_line{2});
        elseif contains(line, 'TileHeight')
            split_line = split(line, ' = ');
            h = str2double(split_line{2});
        end
    end

    if (x==inf || y==inf)
        disp('Could not find X or Y coordinate of the alignment.')
    end
    if (w==inf || h==inf)
        disp('Could not find tile width or height coordinate of the alignment.')
    end
end

%%
function rgb = vals2colormap(vals, colormap, crange)

% Take in a vector of N values and return and return a Nx3 matrix of RGB
% values associated with a given colormap
%
% rgb = AFQ_vals2colormap(vals, [colormap = 'jet'], [crange])
%
% Inputs:
% vals     = A vector of values to map to a colormap or a cell array of
%            vectors of values
% colormap = A matlab colormap. Examples: colormap = 'autumn';
%            colormap = 'jet'; colormap = 'hot';
% crange   = The values to map to the minimum and maximum of the colormap.
%            Defualts to the full range of values in vals.
%
% Outputs:
% rgb      = Nx3 matrix of rgb values mapping each value in vals to the
%            corresponding rgb colors.  If vals is a cell array then rgb
%            will be a cell array of the same length
%
% Example:
% vals = rand(1,100);
% rgb = AFQ_vals2colormap(vals, 'hot');
%
% Copyright Jason D. Yeatman, June 2012

if ~exist('colormap','var') || isempty(colormap)
    colormap = 'jet';
end

%
if ~iscell(vals)
    if ~exist('crange','var') || isempty(crange)
        crange = [min(vals) max(vals)];
    end

    % Generate the colormap
    cmap = eval([colormap '(256)']);
    % Normalize the values to be between 1 and 256
    vals(vals < crange(1)) = crange(1);
    vals(vals > crange(2)) = crange(2);
    valsN = round(((vals - crange(1)) ./ diff(crange)) .* 255)+1;
    
    % Convert any nans to ones
    valsN(isnan(valsN)) = 1;
    % Convert the normalized values to the RGB values of the colormap
    rgb = cmap(valsN, :);

elseif iscell(vals)
    if ~exist('crange','var') || isempty(crange)
        crange = [min(vertcat(vals{:})) max(vertcat(vals{:}))];
    end

    % Generate the colormap
    cmap = eval([colormap '(256)']);

    for ii = 1:length(vals)

        % Normalize the values to be between 1 and 256 for cell ii
        valsN = vals{ii};
        valsN(valsN < crange(1)) = crange(1);
        valsN(valsN > crange(2)) = crange(2);
        valsN = round(((valsN - crange(1)) ./ diff(crange)) .* 255)+1;

        % Convert any nans to ones
        valsN(isnan(valsN)) = 1;

        % Convert the normalized values to the RGB values of the colormap
        rgb{ii} = cmap(valsN, :);
    end
end
return
end