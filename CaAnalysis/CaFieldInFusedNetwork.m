clear all
close all

intial_path = 'M:\tnw\bn\dm\Shared';

% Choose data
% [timelapse_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the timelapse file', intial_path);
% timelapse_path = fullfile(directory, timelapse_file);
% 
segmented_file = uigetfile({'*.tif';'*.png'}, 'Choose the FUSED segmented image', directory);
segmented_path = fullfile(directory, segmented_file);
segmentation = imread(segmented_path);
% background_mask = logical(1-(segmentation>0));
% foreground_mask = ~background_mask;
% 
% rgb_file = uigetfile({'*.tif';'*.png'}, 'Choose the FUSED RGB image.', directory);
% rgb_path = fullfile(directory, rgb_file);
% cell_image = imread(rgb_path);
% 
% network_file = uigetfile('.mat','Choose network file',directory); 
% network_path = fullfile(directory, network_file);
% network = load(network);

[timelapse_file, directory] = uigetfile({'*.tif';'*.png'}, 'Choose the timelapse file', intial_path);
alignment_file = uigetfile('.txt','Choose alignment file',directory); 
alignment_path = fullfile(directory, alignment_file);

alignment_txt = fopen(alignment_path);

%% Read all lines & collect in cell array
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

tile_segmentation = segmentation(y:y+h, x:x+w);
figure()
imshow(tile_segmentation)
