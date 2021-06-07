% Created on 07/06/2021 by Lukas van den Heuvel.
% This code serves as a guideline for reading and plotting the output of a
% Cellpose segmentation.

clear all
close all

%% Load data

% Choose files (RGB image, segmented image and network measurements)
[fused_file,raw_path] = uigetfile('.tif', 'Choose the fused RGB image.');
cd(raw_path)
segmented_file = uigetfile('.tif', 'Choose the segmented image.');
network_file = uigetfile('.mat', 'Choose the Matlab file containing the network measurements.');

% Load the fused image, segmented image and network
disp('Loading data...')
fused = imread(fused_file);
segmented = imread(segmented_file);
network = load(network_file);
G = graph(network.contact_matrix);

%% Load cell measurements
x_nodes = network.centroid1;            % x-coordinates ofcenters of mass 
y_nodes = network.centroid0;            % y-centers of mass measured from the origin
pix_to_um = 6300/length(fused);      	% number of micrometers per pixel
area = network.area * pix_to_um^2;      % area in um^2

%% Social network analysis: normalized betweenness centrality
num_nodes = numnodes(G);
betweenness = 2*centrality(G,'betweenness')/((num_nodes-1)*(num_nodes-2));

%% Plotting
figure()

% Show segmentation
subplot(1,3,1)
rgb = label2rgb(segmented,'jet',[0,0,0],'shuffle'); % randomly relabel to RGB colormap for visualization
imshow(rgb)
title('Segmentation')

% Show fused image with network overlay
subplot(1,3,2)
imshow(fused)
hold on
plot(G,'XData',x_nodes,'YData',y_nodes,'MarkerSize',1,'LineWidth',1,'NodeColor','w','EdgeColor','w')
hold off
title('Network representation')

% Plot the betweenness centrality versus the area of a cell
subplot(1,3,3)
loglog(area, betweenness, '.r')
ylabel('Normalized betweenness centrality')
xlabel('Cell area (\mum^2)')
title('Betweenness versus area')

set(gcf,'Color','w','Units','inches','Position',[9 1 10 3])
