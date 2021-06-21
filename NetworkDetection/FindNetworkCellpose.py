# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:04:40 2021

@author: lukasvandenheu
"""

#%%
import os
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import measure
from scipy import sparse
from scipy.io import savemat

from dialog_helpers import choose_images_and_cellpose_model_with_gui,choose_cellpose_parameters_with_gui
from segment_cellpose_helpers import find_network,cellpose_segment,remove_small_cells,segment_fused_image_with_cellpose,get_image_dimensions

from cellpose import models
use_GPU = models.use_gpu()
print('>>>> GPU activated? %d'%use_GPU)

#%%
def save_metadata_file(metadata_file_path, metadata):
    '''
    This function saves metadata to a txt file inside a well directory.
    
    INPUTS
    ------
    directory (string)
    Path to output directory
    
    metadata (string)
    Metadata to save.
    
    OUTPUT
    ------
    n (bool)
    Is True if the save was succesful, and False otherwise.
    '''
    metadata_file = open(metadata_file_path, 'w')
    n = metadata_file.write(metadata)
    metadata_file.close()
    
    return n

#%%
def load_images_to_preview(img_list):
    '''
    This function loads a set of images that we want to preview.
    If the image is very large (a fused image), we only want to read one image
    (since it takes so much time), and then show different patches from the fused image.
    
    If the image is small enough and we have more of them, we are loading them all
    s.t. the user can preview them.
    
    INPUTS
    ------
    img_list (list of tuples)
    List of paths to images that the user wants to segment. 
    The first entry in each tuple is the path to image, the second is the path
    where we want the output to be stored.
    
    
    OUTPUT
    ------
    preview_images (list of numpy arrays)
    List of images to preview.
    
    divide_in_patches (bool)
    Is true if the first input image is larger than 1200 pixels in height, and
    false otherwise.
    '''
    print('>>>> READING FIRST IMAGE. THIS MAY TAKE SOME TIME...')
    preview_images = [io.imread(img_list[0][0])]
    
    # Check the input dimensions of the image. if the fused image is larger than 1200 pixels in height,
    # we are going to divide it in patches. If it isn't, the images in img_list will be the patch list.
    divide_in_patches = False
    [M,N,C] = get_image_dimensions(preview_images[0])
    
    if (M > 1200): # if the image is large, we are going to divide it in patches.
        divide_in_patches = True
    else:          # if the image is small, we are going to read all images in the folder.
        # Read all images in the folder
        for ii in range(1,len(img_list)):
            if ii==1:
                print('>>>> READING ALL OTHER IMAGES. THIS MAY TAKE SOME TIME...')
            filename = img_list[ii][0]
            img = io.imread(filename)
            [M_new,N_new,C_new] = get_image_dimensions(img)
            # Raise error if the newly read image has different dimensions
            if not([M_new,N_new,C_new] == [M,N,C]):
                raise ValueError("Fatal error: the images to segment must all have the same dimensions!")
            preview_images.append(img)
            
    return preview_images,divide_in_patches
    

#%%
# ----------------------------SPECIFY PARAMETERS-------------------------------

# Cell properties to measure
properties = ['label', 'area', 
              'centroid', 'orientation', 
              'minor_axis_length', 
              'major_axis_length', 
              'eccentricity', 'perimeter']

# Starting directory (for GUI)
initial_dir = r'M:\tnw\bn\dm\Shared' 

# intialize global variable for 'preview' functionality
patch_nr = None 
# Choose images and parameters with TKInter GUI
img_list,model,do_measurements = choose_images_and_cellpose_model_with_gui(initial_dir)
# Load the images you want to previews
preview_images,divide_in_patches = load_images_to_preview(img_list)

[patch_width, patch_height, edge_thickness, similarity_threshold,  
     cell_distance, minimal_cell_size, channels, cell_diameter, num_cpu_cores, 
     parameters_as_string] = choose_cellpose_parameters_with_gui(model,preview_images,divide_in_patches,predict_network=do_measurements)

#%%
# -------------------------------START CODE------------------------------------

# CELLPOSE PARAMETERS
# model_type='cyto' or model_type='nuclei'

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
# channels = [0,0]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus
# channels = [1,3] # IF YOU HAVE R=cytoplasm and B=nucleus

# or if you have different types of channels in each image
# channels = [[2,3], [0,0], [0,0]]

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended) 
# diameter can be a list or a single number for all images

# Model name for output path:
model_name = ""
if (model=='cyto' or model=='nuclei'):
    model_name = model
else:
    model_name = 'custom_model'

# loop through files
print('>>>> PREPARING TO SEGMENT %d IMAGES'%len(img_list))
plt.close('all')
ii = 0
for (filename,output_path) in img_list:

    directory = os.path.split(filename)[0]
    filename_without_extension = os.path.split(filename)[1].split('.')[0]
    
    # Save metadata file
    metadata_file = filename_without_extension + '_cellpose_parameters_'+model_name+'.txt'
    metadata_file_path = os.path.join(output_path, metadata_file)
    saved = save_metadata_file(metadata_file_path, parameters_as_string)
    print('>>>> SAVED METADATA FILE IN ' + output_path + '.')
    
    if (ii>=len(preview_images)): # If the fused image is not yet loaded.
        print('>>>> READING FUSED IMAGE IN ' + output_path + '...')
        fused = io.imread(filename)
    else:
        fused = preview_images[ii]
    
    # Do the cellpose segmentation
    if divide_in_patches:
        segmented = segment_fused_image_with_cellpose(model, fused, cell_diameter, channels,
                                                      edge_thickness=edge_thickness, similarity_threshold=similarity_threshold,
                                                      cell_size_threshold=minimal_cell_size, patch_height=patch_height, patch_width=patch_width,
                                                      num_cpu_cores=num_cpu_cores)
    else:
        segmented = cellpose_segment(model, fused, cell_diameter, channels)
        segmented = remove_small_cells(segmented, minimal_cell_size)
    
    if do_measurements:
        print('>>>> FINDING NETWORK...')
        M,N = segmented.shape # segmented image has M rows and N columns
        
        # Do cell measurements
        measurements = measure.regionprops_table(segmented, properties=properties)
        
        # Find network
        network = find_network(segmented,R=cell_distance)
        sparse_contact_matrix = sparse.lil_matrix(network)
        
        # Store network and cell measurements in dictionary
        mdic = {'contact_matrix': sparse_contact_matrix,'img_size': [M, N]}
        for key,value in measurements.items():
            key = key.replace('-','') # Fields containing '-' are invalid in Matlab
            mdic[key] = value
    
    # Save results
    print('>>>> SAVING OUTPUT...')
    
    output_seg = filename_without_extension + '_cellpose_segmentation_'+model_name+'.tif'
    segmentation_output_path = os.path.join(output_path, output_seg)
    io.imsave(segmentation_output_path, segmented)
    print('>>>> Segmentation is saved succesfully as ' + segmentation_output_path)
    
    if do_measurements:
        output_netw = filename_without_extension + '_network_'+model_name+'.mat'
        network_output_path = os.path.join(output_path, output_netw)
        savemat(network_output_path, mdic)  
        print('>>>> Network is saved succesfully as ' + network_output_path)
        print('\n\n')
    
    ii = ii + 1
