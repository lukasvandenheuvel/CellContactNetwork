# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:21:30 2021

@author: lukasvandenheu
"""


from skimage import io
import numpy as np
import os
from cellpose import models
from skimage import measure
from scipy.io import savemat
from scipy import sparse
from segment_cellpose_helpers import find_network,remove_small_cells,calculate_img_similarity
from dialog_helpers import choose_image_with_gui,choose_cellpose_parameters_with_gui
use_GPU = models.use_gpu()
print('>>>> GPU activated? %d'%use_GPU)

#%%
def find_similarity_with_overlapping_cells(boolean_image, overlapping_image, similarity_threshold):
    '''
    This function finds which cells on the combined_overlap image
    overlap with the boolean image cell_on_overlap.
    '''
    overlapping_cell_values = boolean_image * overlapping_image
    unique = np.sort(np.unique(overlapping_cell_values))
    similiarity_values = []
    
    # loop through cells which overlap with the cell on the egde
    # and calculate similarity
    for overlapping_value in unique[1:]:
        cell_on_mask = (overlapping_image==overlapping_value)
        similarity = calculate_img_similarity(cell_on_mask, boolean_image)
        if similarity > similarity_threshold:
            similiarity_values.append((overlapping_value,similarity))
    
    return similiarity_values

#%%
def equalize_cells_across_timelapse(segmented_timelapse, similarity_threshold):
    '''
    This function takes as input a segmented stack, where each time frame is 
    segmented individually (e.g. with Cellpose).
    The goal of this function is to give a cell which appears on multiple 
    timepoints the same value on each timepoint.
    To do this, we calculate the similarity between a cell on frame f(t) with
    the cells it overlaps with on frame f(t+1).
    
    Parameters
    ----------
    segmented_timelapse (N x M x num_timepoints numpy array)
    Segmented timelapse.
    
    similarity_threshold (double)
    Cells on subsequent timepoints which overlap more then similarity_threshold 
    are given the same value.

    Returns
    -------
    aligned_timelapse (N x M x num_timepoints numpy array)
    Timelapse where cells with sufficient overlap have the same value
    '''
    
    [num_timepoints, M, N] = np.shape(segmented_timelapse)
    
    # initialize an empty timelapse, which will be filled with the aligned cell values
    aligned_timelapse = np.zeros(np.shape(segmented_timelapse))
    # the first frame in aligned_timelapse is equal to that of the timelapse itself
    aligned_timelapse[0,:,:] = segmented_timelapse[0,:,:]
    
    # Loop over timepoints
    for t in range(num_timepoints-1):
        print('Aligning timepoint %d...'%t)
        f0 = aligned_timelapse[t,:,:]       # first frame
        f1 = segmented_timelapse[t+1,:,:]   # second frame
        
        # Get the cells present on the first frame
        unique_cell_values = np.sort(np.unique(f0))
        # Get rid of 0, it is background
        unique_cell_values = unique_cell_values[1:]
        
        # Loop over all cells on first frame
        for nr in unique_cell_values:
            cell = (f0 == nr) # boolean image with True values indicating the cell on first frame
            similiarity_values = find_similarity_with_overlapping_cells(cell, f1, similarity_threshold)
            if (len(similiarity_values) == 0):
                print('Warning: No cell with sufficient similarity found for cell %d'%nr)
                continue
        
            [overlapping_cells, similarity_scores] = zip(*similiarity_values)
            overlapping_cells = np.array(overlapping_cells)
            similarity_scores = np.array(similarity_scores)
        
            if len(overlapping_cells) > 1:
                print('Warning: multiple cells have equal similarity with cell nr %d'%nr)
            
            # Theoretically there should be only one overlapping_cell_nr, but 
            # sometimes there are more (if the cell is split across timepoints).
            # Then, give all overlapping cells the value nr 
            for ol_nr in overlapping_cells: 
                aligned_timelapse[t+1,f1==ol_nr] = nr
            
        
    return aligned_timelapse

#%% 
def init_timelapse_dic(measurements):
    '''
    This function initializes an output dictionary to store the results over time.
    Every key in mdic is a measured property (like 'contact_matrix', 'centroid0', etc.)
    Every value is a list of length num_timepoints, where the elements in the list correspond to the 
    different timepoints.
    
    Input
    -----
    measurements (dict)
    A dictionary with example measurements. Every key in measurements will also
    be a key in mdic.
    
    Output
    ------
    mdic (dict)
    A dictionary with keys 'contac_matrix', 'img_size', 'cell_values_present',
    and all keys present in the measurements dict.
    All values are empty lists.
    '''
    
    mdic = {}
    mdic['contact_matrix'] = []
    mdic['img_size'] = []
    mdic['cell_values_present'] = []
    for key in measurements.keys():
        key = key.replace('-','') # Fields containing '-' are invalid in Matlab
        mdic[key] = []

    return mdic

#%% --------------------------------- INPUTS ----------------------------------
initial_dir = r'M:\tnw\bn\dm\Shared' 

# Choose timelapse with a dialog
title = 'Select RGB timelapse file'
path_to_rgb = choose_image_with_gui(initial_dir,title)
print('Reading timelapse...')
rgb_timelapse = io.imread(path_to_rgb)

# Choose cellpose parameters with a gui
model_name = 'cyto'
first_frame = rgb_timelapse[0,:,:,:]
do_measurements = True
[_, _, _, _, similarity_threshold,  
     cell_distance, minimal_cell_size, channels, cell_diameter, num_cpu_cores, 
     parameters_as_string] = choose_cellpose_parameters_with_gui(model_name,first_frame,do_measurements)

# Measurement properties
properties = ['label', 'area', 
              'centroid', 'orientation', 
              'minor_axis_length', 
              'major_axis_length', 
              'eccentricity', 'perimeter']

#%% ------------------------ SEGMENT TIMEFRAMES ------------------------------- 
directory = os.path.split(path_to_rgb)[0]
file_name = os.path.split(path_to_rgb)[1].split('.')[0]
[num_timepoints, M, N, C] = np.shape(rgb_timelapse)

cellpose_model = models.Cellpose(gpu=use_GPU, model_type=model_name)
segmented_timelapse = np.zeros((num_timepoints, M, N))

# Segment individual timeframes with cellpose
for t in range(num_timepoints):
    patch = rgb_timelapse[t,:,:,:]
    mask, flows, styles, diams = cellpose_model.eval(patch, diameter=cell_diameter, flow_threshold=None, channels=channels)
    filtered_mask = remove_small_cells(mask, minimal_cell_size)
    segmented_timelapse[t,:,:] = filtered_mask
    print('Segmented timepoint %d'%t)

#%% ------------------------ ALIGN TIMEFRAMES --------------------------------

# Give cells on subsequent timepoints which overlap the same value
segmented_timelapse = np.asarray(segmented_timelapse, dtype='uint64')
aligned_timelapse = equalize_cells_across_timelapse(segmented_timelapse, similarity_threshold)

#%% ---------------- INITIALIZE RESULTS DICTIONARY ---------------------------

# do an example measurement, to be used to initialise an output dictionary
f0 = np.asarray(aligned_timelapse[0,:,:], dtype='uint16')
num_cells_on_first_timeframe = np.max(f0)
measurements = measure.regionprops_table(f0, properties=properties)

# Initialise a dictionary with empty lists, to store the results per timepoint
timelapse_dic = init_timelapse_dic(measurements)

#%% -------------------- MEASURE AND FIND NETWORK -----------------------------

# Find measurements and network on each timepoint
for t in range(num_timepoints):

    print('Finding network on timepoint %d...'%t)
    segmented = np.asarray(aligned_timelapse[t,:,:], dtype='uint16')
    cell_values_present = np.sort(np.unique(segmented))
    cell_values_present = cell_values_present[1:] # remove 0
    
    # Do cell measurements
    measurements = measure.regionprops_table(segmented, properties=properties)
    
    # Find network
    network = find_network(segmented,max_num_cells=num_cells_on_first_timeframe,R=cell_distance)
    sparse_contact_matrix = sparse.lil_matrix(network)
    
    # Store network and cell measurements in dictionary
    timelapse_dic['contact_matrix'].append(sparse_contact_matrix)
    timelapse_dic['img_size'].append((M, N))                     
    timelapse_dic['cell_values_present'].append(cell_values_present)        
    for key,value in measurements.items():
        key = key.replace('-','') # Fields containing '-' are invalid in Matlab
        timelapse_dic[key].append(value)
    
#%% --------------------------- SAVE RESULTS ---------------------------------

# Save the segmented timelapse and the dictionary
print('Saving results...')
output_file_segmentation = file_name + '_cellpose_segmentation_'+model_name+'.tif'
output_file_network = file_name + '_network_'+model_name+'.mat'

io.imsave(os.path.join(directory, output_file_segmentation), aligned_timelapse)
savemat(os.path.join(directory, output_file_network), timelapse_dic)
