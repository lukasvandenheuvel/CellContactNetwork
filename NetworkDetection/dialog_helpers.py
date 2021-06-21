# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:54:53 2021

@author: lukasvandenheu
"""
import os
import numpy as np
import sys
import multiprocessing
import tkinter as tk
import tkinter.filedialog
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from skimage import measure
from skimage.draw import line, circle

from segment_cellpose_helpers import find_network,cellpose_segment,enlarge_fused_image,fused_to_patches,remove_small_cells,get_image_dimensions
from cellpose import utils
from scipy import sparse

# Initialise global variables
patch_nr = None
patch_list = []
outline_color = [255,255,255]
cmap = None

#%%
def plot_outlines(mask, img, rgb_color, cmap):
    '''
    This function shows the outlines of a segmentation on an RGB image.
    The rgb_color is a list with 3 entries, e.g. [0,255,0] for green.
    '''
    outlines = utils.masks_to_outlines(mask)
    outX, outY = np.nonzero(outlines)
    rgb_color = rgb_color / np.max(rgb_color)
    imgout= img / np.max(img)
    imgout[outX, outY] = np.array(rgb_color)
    plt.imshow(imgout, cmap=cmap)

#%%
def get_display_size():
    '''
    This function outputs the height and width of the computer display in pixels.
    '''
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    height = root.winfo_screenheight()
    width = root.winfo_screenwidth()
    root.destroy()
    return height, width

#%%
def close_program(gui, exit_code=False):
    '''
    This function closes all open matplotlib.pyplot windows.
    It also closes the TKinter object 'gui'.
    If exit_code==True, it also exits the code.
    '''
    plt.close('all')
    gui.destroy()
    if exit_code:
        sys.exit("User terminated the program")
        
#%%
def choose_random_patch_nr(patch_list):
    '''
    This function takes as input a list of patches. It outputs a random nr
    (between 0 and length(patch_list)) which is the index of some patch with a
    maximum pixel intensity larger than 10.
    '''
    num_patches = len(patch_list)
    nr = np.random.randint(0,num_patches)
    while np.max(patch_list[nr]) < 10:  # max intensity of patch must be larger than pixel value 10
        nr = np.random.randint(0,num_patches)
    return nr

#%%
def find_channels(model, cyto_ch, nucleo_ch):
    '''
    This function converts channel colors e.g. 'gray', 'R', 'G', 'B' to a list
    called channels. The first entry in the list is the cytoplasm channel,
    the second is the nucleus channel.
    Example:
    channels=[2,3] if you have cyto_ch='G' and nucleo_ch='B'.
    
    INPUTS
    ------
        model (str)
        Can be 'cyto' to use Cellpose pre-trained cyto model, 'nuclei' to use 
        Cellpose pre-trained nucleus model, or a path to a file containing the
        weights of a custom-trained model. An example of such a file is  
        "cellpose_residual_on_style_on_concatenation_off_Cellpose_2021_05_04.236206"
        
        cyto_ch_str (str)
        Indicates the color of the cytoplasm. Can be 'gray', 'R', 'G', 'B' or 'None'.
        
        nucleo_ch_str (str)
        Indicates the color of the cytoplasm. Can be 'gray', 'R', 'G', 'B' or 'None'.
    
    OUTPUT
    ------
        channels (list of 2 integers)
        First channel is cyto, second is nucleus. Example:
        channels=[2,3] if you have G=cytoplasm and B=nucleus 
        
    '''
    # Check if cyto and nucleo colors are entered correctly
    options = ['gray', 'R', 'G', 'B', 'None']
    if not(cyto_ch in options) or not(nucleo_ch in options):
        raise ValueError("You entered an invalid color for the nucleus or cytoplasm channel. Choose from 'gray', 'R', 'G', 'B' or 'None'.")
    
    # Convert cyto and nucleo colors to a list with 2 channels (first channel is 
    # for cytoplasm, second channel for nuclei).
    cyto_channel = options.index(cyto_ch)
    cyto_channel = (0 if cyto_channel==4 else cyto_channel) # if cyto_ch_str was 'None', set cyto_channel to 0.
    nucleo_channel = options.index(nucleo_ch)
    nucleo_channel = (0 if nucleo_channel==4 else nucleo_channel) # if nucleo_ch_str was 'None', set nucle0_channel to 0.
    channels = [cyto_channel, nucleo_channel]
    
    # If we use the nucleus model, the first channel should be the nucleus channel
    # and the second channel should be 0
    # (the cyto channel should not be used in that case)
    if model=='nuclei':
        print('Using the nuclei model.')
        channels = [nucleo_channel,0]
    
    return channels

#%%
def draw_network(img, A, x_nodes, y_nodes):
    '''
    Function draws the network as nodes and edges on the image.
    '''
    # Make A a sparse network
    A = sparse.lil_matrix(A)
    
    # Create normalized RGB output image:
    N,M,C = get_image_dimensions(img)
    color = outline_color / np.max(outline_color) # list of 1 values with the same length as the number of channels ([1] for gray, [1,1,1] for RGB)
    output = img / np.max(img)
    radius = 5 # size of nodes

    # Loop over nodes:
    for x, y in zip(x_nodes, y_nodes):
        c = int(x)
        r = int(y)
        # get indices of the circle
        rr, cc = circle(r, c, radius)
        # make sure the indices are not larger than the image edge
        rr = np.minimum(N-1, rr)
        cc = np.minimum(M-1, cc)
        
        # Draw node as white circle:
        output[rr, cc, :] = color

    # Loop over edges:
    i_list, j_list = A.nonzero()
    for i,j in zip(i_list, j_list):
        xi, yi = x_nodes[i], y_nodes[i]  # x,y position node i
        xj, yj = x_nodes[j], y_nodes[j]  # x,y position node j
        ci = int(xi)                     # column position of node i
        cj = int(xj)                     # column position of node j
        ri = int(yi)                     # row position of node i
        rj = int(yj)                     # row position of node j
        rr, cc = line(ri,ci,rj,cj)

        # Draw edge as while line:
        output[rr, cc, :] = color

    return output

#%% 
def show_one_segmentation(model_type, patch, diameter, channels, cell_size_threshold, cell_distance, predict_network):
    '''
    This function shows the user the result of a segmentation. It is called by
    the 'preview_segmentation' function, which in turn is called if the user
    pushes the "Preview" button.
    
    INPUTS
    -----
        model_type (str)
        Can be 'cyto' to use Cellpose pre-trained cyto model, 'nuclei' to use 
        Cellpose pre-trained nucleus model, or a path to a file containing the
        weights of a custom-trained model. An example of such a file is  
        "cellpose_residual_on_style_on_concatenation_off_Cellpose_2021_05_04.236206"
        
        patch (patch_height x patch_width x C numpy array)
        One patch to show.
        
        diameter (float)
        Cellpose cell diameter. If the model is custom-built, then diameter=None.
        
        channels (list of 2 integers)
        First channel is cyto, second is nucleus. Example:
        channels=[2,3] if you have G=cytoplasm and B=nucleus
    '''
    global outline_color
    global cmap
    
    screen_height,screen_width = get_display_size()
    dpi = 100
    [m,n,C] = get_image_dimensions(patch)
        
    fig_width = int (2 * screen_width / 3) 
    fig_height = int(2 * screen_height / 3 )
    fig_x_pos = int( screen_width / 3 )
    fig_y_pos = 50
    
    # Do cellpose segmentation
    mask = cellpose_segment(model_type, patch, diameter, channels)
    filtered_mask = remove_small_cells(mask, cell_size_threshold)
    
    if predict_network:
        # Measure x and y coordinates of nodes
        measurements = measure.regionprops_table(filtered_mask, properties=['centroid'])
        y_nodes = measurements['centroid-0']
        x_nodes = measurements['centroid-1']
        
        # Find network
        network = find_network(filtered_mask, R=cell_distance)
        network_img = draw_network(patch, network, x_nodes, y_nodes)
    
    #f = plt.figure(figsize=[fig_width,fig_height])
    f = plt.figure()    
    #f.canvas.manager.window.wm_geometry("+%d+%d" % (fig_x_pos, fig_y_pos))
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(fig_x_pos, fig_y_pos, fig_width, fig_height)

    plt.subplot(1,2,1)
    plot_outlines(filtered_mask, patch, outline_color, cmap)
    plt.title('Segmentation outlines')
    
    plt.subplot(1,2,2)
    if predict_network:
        plt.imshow(network_img, cmap=cmap)
        plt.title('Network prediction')
    else:
        plt.imshow(filtered_mask)
        plt.title('Segmentation masks')
        
    if (model_type=='cyto' or model_type=='nuclei'):
        plt.suptitle('Diameter = %d'%diameter)
    f.show()
    
#%%
def preview_segmentation(model, preview_images, divide_in_patches, diameter, 
                         patch_height, patch_width, cyto_ch_str, nucleo_ch_str,
                         cell_size_threshold, cell_distance, predict_network):
    '''
    This function is called if the user clicks the 'preview' button.
    It divides the fused image into patches (if divide_in_patches is True),
    chooses a random patch, segments this patch and shows it to the user.
    
    INPUTS
    -------
        model (str)
        Can be 'cyto' to use Cellpose pre-trained cyto model, 'nuclei' to use 
        Cellpose pre-trained nucleus model, or a path to a file containing the
        weights of a custom-trained model. An example of such a file is  
        "cellpose_residual_on_style_on_concatenation_off_Cellpose_2021_05_04.236206"
    
        preview_images (list of MxNxC numpy arrays)
        Image(s) to segment.
        
        diameter (float)
        Cellpose cell diameter. If the model is custom-built, then diameter=None.
        
        patch_width, patch_height (int)
        Width and height of a single patch in pixels.
        
        cyto_ch_str (str)
        Indicates the color of the cytoplasm. Can be 'gray', 'R', 'G', 'B' or 'None'.
        
        nucleo_ch_str (str)
        Indicates the color of the cytoplasm. Can be 'gray', 'R', 'G', 'B' or 'None'.
        
        cell_size_threshold (int)
        Minimal area of a cell in pixels.
        
        cell_distance (int)
        Maximal number of pixels that separates two neighbouring cells
    '''
    
    # Global variables: these are changed in this function, and thereby changed
    # in the whole script!!
    # I don't like global variables, but right now we have little choice because
    # we want the variables patch_nr and patch_list to change when the user clicks
    # the button "preview".
    global patch_nr
    global patch_list
    global outline_color
    global cmap
    
    screen_height,screen_width = get_display_size()
    channels = find_channels(model, cyto_ch_str, nucleo_ch_str)
  
    # if images are grayscale, unsqueeze them
    [M,N,C] = get_image_dimensions(preview_images[0])
    if (C==1): 
        outline_color = [255]
        cmap = 'gray'
        for ii,img in enumerate(preview_images):
            preview_images[ii] = np.reshape(img, (M,N,1))
        
    overlap = int(patch_height/2)

    # Divide the fused image into patches if the user asked to do so.
    if divide_in_patches:
        fused = preview_images[0]
        [fused, patch_locations] = enlarge_fused_image(fused, patch_height=patch_height, patch_width=patch_width, overlap=overlap)
        new_patch_list = fused_to_patches(fused, patch_locations, patch_height=patch_height, patch_width=patch_width)
    else:
        new_patch_list = preview_images
    
    # If the new patch_list differs from the previous one, we need to choose a new patch_nr.
    # This is for example the case when the user clicks the 'preview' button for
    # the first time, or if the user changed the width / height of the patch.
    if not(np.array_equal(new_patch_list, patch_list)):
        patch_nr = choose_random_patch_nr(new_patch_list) # because patch_nr is a global variable, it will now be updated throughout the whole script!

    patch_list = new_patch_list # because patch_list is a global variable, it will now be updated throughout the whole script!
    
    # Show the segmentation.
    show_one_segmentation(model, patch_list[patch_nr], diameter, channels, cell_size_threshold, cell_distance, predict_network)
    
    # Show a button to change the patch number
    master = tk.Tk()
    x_left = int( screen_width - 300 )
    y_top = int( screen_height - 300 )
    master.geometry("{}x{}+{}+{}".format(200,100,x_left,y_top))
    
    B1 = tk.Button(master, text='Show different patch', command=lambda:change_patch_nr(model, patch_list, diameter, channels, cell_size_threshold, cell_distance, predict_network), bg="yellow")
    B1.grid(row=1, sticky=tk.W, pady=4)
    
    B2 = tk.Button(master, text='Change outline color / cmap', command=lambda:change_outline_color(model, patch_list, diameter, channels, cell_size_threshold, cell_distance, predict_network), bg="yellow")
    B2.grid(row=2, sticky=tk.W, pady=4)
    
    B3 = tk.Button(master, text='Close', command=lambda:close_program(master), bg="red")
    B3.grid(row=3, sticky=tk.W, pady=4)
    
    master.mainloop()

#%%
def change_patch_nr(model, patch_list, diameter, channels, cell_size_threshold, cell_distance, predict_network):
    '''
    This function randomly changes the patch number, which indicates which patch in the
    patch_list to show to the user.
    Then, it shows the new patch.
    '''
    global patch_nr
    new_patch_nr = choose_random_patch_nr(patch_list) 
    # Make sure the new patch nr is not the same as the old one, unless there is only one patch
    while (new_patch_nr==patch_nr and len(patch_list)>1):
        new_patch_nr = choose_random_patch_nr(patch_list)
    patch_nr = new_patch_nr # because patch_nr is a global variable, it will now be updated throughout the whole script!
    show_one_segmentation(model, patch_list[patch_nr], diameter, channels, cell_size_threshold, cell_distance, predict_network)

#%%
def change_outline_color(model, patch_list, diameter, channels, cell_size_threshold, cell_distance, predict_network):
    
    global outline_color
    global cmap
    if len(outline_color)==1: #grayscale image
        possible_cmaps = ['hot','gray',None,'Greens']
        current_index = possible_cmaps.index(cmap)
        new_index = (0 if current_index==3 else current_index+1) # move one further, or go back to start
        cmap = possible_cmaps[new_index]
    
    else: # RGG image
        possible_colors = [[255,0,0], [0,255,0],[0,0,255],[255,255,255]]
        current_index = possible_colors.index(outline_color)
        new_index = (0 if current_index==3 else current_index+1) # move one further, or go back to start
        outline_color = possible_colors[new_index]
    show_one_segmentation(model, patch_list[patch_nr], diameter, channels, cell_size_threshold, cell_distance, predict_network)
    
    
#%%
def choose_cellpose_parameters_with_gui(model,preview_images,divide_in_patches,predict_network=True):
    '''
    This function starts up a TKInter graphical user interface to choose all parameters.
    
    INPUTS
    ------
        model (str)
        Can be 'cyto' to use Cellpose pre-trained cyto model, 'nuclei' to use 
        Cellpose pre-trained nucleus model, or a path to a file containing the
        weights of a custom-trained model. An example of such a file is  
        "cellpose_residual_on_style_on_concatenation_off_Cellpose_2021_05_04.236206"
        
        preview_images ( list of MxNxC numpy arrays )
        Image(s) to segment.
    
    OUTPUTS
    -------
        divide_in_patches_var (bool)
        True if the user wants to divide the image in patches, and False otherwise.
        
        patch_width, patch_height (int)
        Width and height of a single patch in pixels.
        
        edge_thickness (int)
        Size of the edge region.
        
        similarity_threshold (int)
        Overlapping cells which are more similar than similarity_threshold are merged.
        
        cell_distance (int)
        Maximal number of pixels that separates two neighbouring cells
        
        minimal_cell_size (int)
        Minimal area of a cell in pixels.
        
        channels (list of 2 integers)
        First channel is cyto, second is nucleus. Example:
        channels=[2,3] if you have G=cytoplasm and B=nucleus
        
        diameter (float)
        Cellpose cell diameter. If the model is custom-built, then diameter=None.
        
        num_cpu_cores (int)
        Number of CPU cores used to calculate the overlap between patches.
        
        metadata (str)
        Summarizes all of the above parameters.
    '''
    
    # Global variables: these are changed in this function, and thereby changed
    # in the whole script!!
    # I don't like global variables, but right now we have little choice because
    # we want the variables patch_nr and patch_list to change when the user clicks
    # the button "preview".
    global patch_nr
    
    [M,N,C] = get_image_dimensions(preview_images[0])
    
    gui = tk.Tk()
    gui.geometry("{}x{}+{}+{}".format(800,650,50,50))
    
    # SPECIFY PARAMETERS #####################################################
    ypos = 10 # initial y position of text on dialog
    tk.Label(gui, text="The selected image consists of %dx%d pixels, and has %d channels."%(M,N,C)).place(x=10,y=ypos)
    
    # Parameters specific to calculation of overlap between patches:
    if divide_in_patches:
        
        ypos = ypos + 20 
        tk.Label(gui, text="The image is large, and will therefore be processed in patches.").place(x=10,y=ypos)
        ypos = ypos + 30 
        tk.Label(gui, text="PATCH PARAMETERS --------------------------------------").place(x=10,y=ypos)

        ypos = ypos + 30
        patch_width = tk.IntVar(value=1024)
        tk.Label(gui, text="Patch width:").place(x=10,y=ypos)
        tk.Label(gui, text="(pixels)").place(x=300,y=ypos)
        entry_width = tk.Entry(gui, textvariable=patch_width)
        entry_width.grid(row=0, sticky=tk.W)
        entry_width.place(x=200,y=ypos,width=90)
        
        ypos = ypos + 30
        patch_height = tk.IntVar(value=1024)
        tk.Label(gui, text="Patch height:").place(x=10,y=ypos)
        tk.Label(gui, text="(pixels)").place(x=300,y=ypos)
        entry_height = tk.Entry(gui, text='Patch height:', textvariable=patch_height)
        entry_height.grid(row=0, sticky=tk.W)
        entry_height.place(x=200,y=ypos,width=90)
        
        ypos = ypos + 30
        edge_thickness = tk.IntVar(value=60)
        tk.Label(gui, text="Size of edge region:").place(x=10,y=ypos)
        tk.Label(gui, text="(pixels)").place(x=300,y=ypos)
        entry_thickness = tk.Entry(gui, textvariable=edge_thickness)
        entry_thickness.grid(row=0, sticky=tk.W)
        entry_thickness.place(x=200,y=ypos,width=90)
        
        ypos = ypos + 30
        similarity_threshold = tk.DoubleVar(value=0.7)
        tk.Label(gui, text="Cell similarity threshold:").place(x=10,y=ypos)
        tk.Label(gui, text="(cells with sufficient similarity are merged)").place(x=300,y=ypos)
        entry_similarity = tk.Entry(gui, textvariable=similarity_threshold)
        entry_similarity.grid(row=0, sticky=tk.W)
        entry_similarity.place(x=200,y=ypos,width=90)
        
        ypos = ypos + 30
        num_cores = tk.IntVar(value=0)
        tk.Label(gui, text="Number of CPU cores:").place(x=10,y=ypos)
        tk.Label(gui, text="(set to 0 to leave 2 cores available and use the rest)").place(x=300,y=ypos)
        entry_cores = tk.Entry(gui, textvariable=num_cores)
        entry_cores.grid(row=0, sticky=tk.W)
        entry_cores.place(x=200,y=ypos,width=90)
        
    else:
        patch_width = tk.IntVar(value=None)
        patch_height = tk.IntVar(value=None)
        edge_thickness = tk.IntVar(value=None)
        similarity_threshold = tk.DoubleVar(value=None)
        num_cores = tk.IntVar(value=None)
    
    # Parameters which are always necessary
    ypos = ypos + 30 
    tk.Label(gui, text="CHANNELS AND CELL SIZE -------------------------------").place(x=10,y=ypos)
    ypos = ypos + 30
    # The default strings for nucleus and cytoplasm channels depend on the model,
    # and on the image type (gray or RGB)
    if (C==1) and (model=='nuclei'): # grayscale image with nucleus model
        default_cyto_str = "None"
        default_nucelo_str = "gray"
    elif (C==1): # grayscale image with no matter what model
        default_cyto_str = "gray"
        default_nucelo_str = "None"
    else: # RGB image
        default_cyto_str = "R"
        default_nucelo_str = "B"

    cyto_ch_str = tk.StringVar(value=default_cyto_str)
    tk.Label(gui, text="Cytoplasm color:").place(x=10,y=ypos)
    tk.Label(gui, text="(Choose 'gray', 'R', 'G', 'B' or 'None')").place(x=300,y=ypos)
    entry_cyto_ch = tk.Entry(gui, textvariable=cyto_ch_str)
    entry_cyto_ch.grid(row=0, sticky=tk.W)
    entry_cyto_ch.place(x=200,y=ypos,width=90)
    
    ypos = ypos + 30
    nucleo_ch_str = tk.StringVar(value=default_nucelo_str)
    tk.Label(gui, text="Nucleus color:").place(x=10,y=ypos)
    tk.Label(gui, text="(Choose 'gray', 'R', 'G', 'B' or 'None')").place(x=300,y=ypos)
    entry_nucleo_ch = tk.Entry(gui, textvariable=nucleo_ch_str)
    entry_nucleo_ch.grid(row=0, sticky=tk.W)
    entry_nucleo_ch.place(x=200,y=ypos,width=90)
    
    ypos = ypos + 30
    minimal_cell_size = tk.IntVar(value=200)
    tk.Label(gui, text="Minimal cell area:").place(x=10,y=ypos)
    tk.Label(gui, text="(pixels)").place(x=300,y=ypos)
    entry_minsize = tk.Entry(gui, textvariable=minimal_cell_size)
    entry_minsize.grid(row=0, sticky=tk.W)
    entry_minsize.place(x=200,y=ypos,width=90)
   
    # Parameter specific to Cellpose model (diameter)
    if (model=='cyto' or model=='nuclei'):
        
        ypos = ypos + 30 
        tk.Label(gui, text="CELLPOSE DIAMETER -------------------------------------").place(x=10,y=ypos)
    
        ypos = ypos + 30
        diameter_var = tk.IntVar(value=60)
        tk.Label(gui, text="Cellpose cell diameter:").place(x=10,y=ypos)
        tk.Label(gui, text="(pixels)").place(x=300,y=ypos)
        entry_diameter = tk.Entry(gui, textvariable=diameter_var)
        entry_diameter.grid(row=0, sticky=tk.W)
        entry_diameter.place(x=200,y=ypos,width=90)
    else:
        diameter_var = tk.IntVar(value=None)
    
    
    # Parameters specific to network detection
    
    if predict_network:
        
        ypos = ypos + 30 
        tk.Label(gui, text="NETWORK DETECTION PARAMETERS ---------------------------").place(x=10,y=ypos)
        
        ypos = ypos + 30
        cell_distance_var = tk.IntVar(value=8)
        tk.Label(gui, text="Max distance separating cells:").place(x=10,y=ypos)
        tk.Label(gui, text="(pixels)").place(x=300,y=ypos)
        entry_distance = tk.Entry(gui, textvariable=cell_distance_var)
        entry_distance.grid(row=0, sticky=tk.W)
        entry_distance.place(x=200,y=ypos,width=90)
    else:
        cell_distance_var = tk.IntVar(value=None)
    
    
    # CREATE BUTTONS ##########################################################
    ypos = ypos + 50
    B_continue = tk.Button(gui, text='Continue', command=gui.destroy, bg="green")
    B_continue.grid(row=2, sticky=tk.W, pady=4)
    B_continue.place(x=10,y=ypos)
    
    B_preview = tk.Button(gui, text='Preview', command=lambda:preview_segmentation(model, preview_images, divide_in_patches, diameter_var.get(), 
                                                                                   patch_height.get(), patch_width.get(),cyto_ch_str.get(),nucleo_ch_str.get(),
                                                                                   minimal_cell_size.get(), cell_distance_var.get(), predict_network), bg="yellow")
    B_preview.grid(row=2, sticky=tk.W, pady=4)
    B_preview.place(x=110,y=ypos)
    
    B_close = tk.Button(gui, text='Close & quit', command=lambda:close_program(gui,exit_code=True), bg="red")
    B_close.grid(row=2, sticky=tk.W, pady=4)
    B_close.place(x=210,y=ypos)
    
    gui.mainloop()
    
    channels = find_channels(model, cyto_ch_str.get(), nucleo_ch_str.get())
    
    # Number of CPU cores
    if (num_cores.get() == 0) or (num_cores.get() >= multiprocessing.cpu_count()):
        num_cpu_cores = multiprocessing.cpu_count() - 2 
    else:
        num_cpu_cores = num_cores.get()
    
    # Get parameters as a string (to save metadata file)
    metadata = ""
    metadata = metadata + "DivideInPatches = " + str(divide_in_patches) + "\n"
    metadata = metadata + "CellposePatchWidth = " + str(patch_width.get()) + "\n"
    metadata = metadata + "CellposePatchHeight = " + str(patch_height.get()) + "\n"
    metadata = metadata + "CellposeModel = " + model + "\n"
    metadata = metadata + "EdgeRegionThickness = " + str(edge_thickness.get()) + "\n"
    metadata = metadata + "CellSimilarityThreshold = " + str(similarity_threshold.get()) + "\n"
    metadata = metadata + "DistanceSeparatingCells = " + str(cell_distance_var.get()) + "\n"
    metadata = metadata + "MinimalCellSize = " + str(minimal_cell_size.get()) + "\n"
    metadata = metadata + "CellposeChannels = " + str(channels) + "\n"
    metadata = metadata + "CellposeDiameter = " + str(diameter_var.get()) + "\n"
    metadata = metadata + "NumberOfCpuCores = " + str(num_cpu_cores) + "\n"
    
    return [patch_width.get(), patch_height.get(), edge_thickness.get(),
            similarity_threshold.get(), cell_distance_var.get(), 
            minimal_cell_size.get(), channels,
            diameter_var.get(), num_cpu_cores, metadata]

#%%
def get_file_name_from_format(filename_format, well):
    '''
    construct file name based on the user-specified filename format.
    Example: if the user entered the format '{WWW}_fused_RGB.tif',
    then file_format = ['', '_fused_RGB.tif']. We now need to convert
    that into B02_fused_RGB.tif, which we do by looping over the entries
    in the file_format list.
    '''
    file_format = filename_format.split('{WWW}')
    file = file_format[0] # init file name with the first entry in the file_format list
    for ii in range(1,len(file_format)):
        file = file + well + file_format[ii]
    
    return file

#%%
def choose_images_and_cellpose_model_with_gui(initial_dir):
    '''
    This function starts up a tkinter dialog. It allows the user to choose 
    the images to process and the model to use.
    
    INPUT
    ------
        initial_dir (str)
        Path to the initial directory shown if you start up the TKinter file chooser.
    
    OUTPUTS
    -------
        img_list (list of tuple)
        First entry of each tuple is the path to the image. Second entry is the 
        path to the output directory.
    
        model (str)
        Can be 'cyto' to use Cellpose pre-trained cyto model, 'nuclei' to use 
        Cellpose pre-trained nucleus model, or a path to a file containing the
        weights of a custom-trained model. An example of such a file is  
        "cellpose_residual_on_style_on_concatenation_off_Cellpose_2021_05_04.236206"
        
        do_measurements (bool)
        True if the user wants to do measurements on the predictions (like
        finding the area, positions and the cell-contact network) and false otherwise.
    '''

    # find out whether the user wants one file processed,
    # or multiple files.    
    master = tk.Tk()
    tk.Label(master, text="1. How many images do you want to process?").grid(row=0, sticky=tk.W)
    one_img = tk.IntVar()
    tk.Checkbutton(master, text="Just one", variable=one_img).grid(row=1, sticky=tk.W)
    multiple_imgs = tk.IntVar()
    tk.Checkbutton(master, text="Multiple in one folder", variable=multiple_imgs).grid(row=2, sticky=tk.W)
    multiple_folders = tk.IntVar()
    tk.Checkbutton(master, text="Multiple in seperate well folders", variable=multiple_folders).grid(row=3, sticky=tk.W)
    
    # find out whether the user want to use the pre-trained Cellpose model, 
    # or a self-trained model.
    tk.Label(master, text="2. Which model do you want to use?").grid(row=5, sticky=tk.W)
    cyto_model = tk.IntVar()
    tk.Checkbutton(master, text="Cellpose cyto model (pre-trained)", variable=cyto_model).grid(row=6, sticky=tk.W)
    nucleo_model = tk.IntVar()
    tk.Checkbutton(master, text="Cellpose nucleus model (pre-trained)", variable=nucleo_model).grid(row=7, sticky=tk.W)
    self_trained = tk.IntVar()
    tk.Checkbutton(master, text="Self-trained model", variable=self_trained).grid(row=8, sticky=tk.W)
    
    # find out whether the user wants to do measurements or not
    tk.Label(master, text="3. Do you want to do measurements? (cell positions, area, cell-contact network, etc.)").grid(row=9, sticky=tk.W)
    do_measure_var = tk.IntVar()
    do_measure_button = tk.Checkbutton(master, text="Yes", variable=do_measure_var)
    do_measure_button.grid(row=10, sticky=tk.W)
    do_measure_button.select() # true by default
    
    B = tk.Button(master, text='Continue', command=master.destroy, bg="green")
    B.grid(row=11, sticky=tk.W, pady=7)
    tk.mainloop()
    
    # FILES TO PROCESS ########################################################
    img_list = []
    
    # Check if the user checked one box only
    num_checked_boxes = np.sum([one_img.get(), multiple_imgs.get(), multiple_folders.get()])
    if (num_checked_boxes != 1):
         raise ValueError('Please choose exactly one option: how many images do you want to process?.')
    
    # PROCESS ONE FILE ONLY
    if one_img.get():  # user chose to process only 1 file
        root = tk.Tk()
        root.filename = tk.filedialog.askopenfilename(initialdir = initial_dir,title = "Select image",filetypes = [("tif files",".tif")])
        img_list.append((root.filename, os.path.split(root.filename)[0]))
        root.destroy() # close GUI
    
    # PROCESS MULTIPLE IMAGES IN ONE FOLDER
    if multiple_imgs.get():
        root = tk.Tk()
        root.filename = tk.filedialog.askdirectory(initialdir = initial_dir, title = "Select directory where the images are")
        root.destroy() # close GUI
        
        out = tk.Tk()
        out.filename = tk.filedialog.askdirectory(initialdir = initial_dir, title = "Select output directory")
        out.destroy() # close GUI
        
        # Specify filenames
        file_chooser = tk.Tk()
        file_chooser.geometry("600x200")
        
        ypos = 10
        tk.Label(file_chooser, text='Filename must include:').place(x=10,y=ypos)
        
        ypos = ypos + 30
        include_var = tk.StringVar(value='_fused')
        entry = tk.Entry(file_chooser, textvariable=include_var)
        entry.grid(row=1, sticky=tk.W)
        entry.place(x=10,y=ypos,width=400)
        
        ypos = ypos + 50
        B_continue = tk.Button(file_chooser, text='Continue', command=file_chooser.destroy)
        B_continue.grid(row=2, sticky=tk.W, pady=4)
        B_continue.place(x=10,y=ypos)
        
        B_close = tk.Button(file_chooser, text='Close & quit', command=lambda:close_program(file_chooser,exit_code=True))
        B_close.grid(row=2, sticky=tk.W, pady=4)
        B_close.place(x=110,y=ypos)
        file_chooser.mainloop()
        
        # Make a list of files to be included
        for file in os.listdir(root.filename):
            if include_var.get() in file:
                image_path = os.path.join(root.filename, file)
                img_list.append((image_path, out.filename))
    
    # PROCESS MULTIPLE IMAGES IN WELL FOLDERS
    if multiple_folders.get():
        root = tk.Tk()
        root.filename = tk.filedialog.askdirectory(initialdir = initial_dir, title = "Select root directory (where the well folders are)")
        root.destroy() # close GUI
        
        # Specify wells
        well_chooser = tk.Tk()
        well_chooser.geometry("600x350")
        
        ypos = 10
        tk.Label(well_chooser, text='Files must be stored in (subfolders of) well folders, e.g. "B02", "B03".').place(x=10,y=ypos)
        
        ypos = ypos + 30
        wellstr = tk.StringVar(value='Enter the wells you want to process, separated by commas.')
        entry = tk.Entry(well_chooser, textvariable=wellstr)
        entry.grid(row=1, sticky=tk.W)
        entry.place(x=10,y=ypos,width=400)
        
        tk.Label(well_chooser, text='OR').place(x=420,y=ypos)
        allwells = tk.IntVar()
        allwellbutton = tk.Checkbutton(well_chooser, text='Process all wells in root', variable=allwells)
        allwellbutton.grid(row=1, sticky=tk.W)
        allwellbutton.place(x=450,y=ypos)
        
        ypos = ypos + 50
        tk.Label(well_chooser, text='Folder name format: ({WWW} is the well name, e.g. "B02".').place(x=10,y=ypos)
        ypos = ypos + 20
        tk.Label(well_chooser, text='Example: {WWW}\CaImaging if the images are in a subfolder B02\CaImaging.').place(x=10,y=ypos)
        ypos = ypos + 30
        foldername_format_var = tk.StringVar(value='{WWW}')
        entry = tk.Entry(well_chooser, textvariable=foldername_format_var)
        entry.grid(row=1, sticky=tk.W)
        entry.place(x=10,y=ypos,width=400)
        
        ypos = ypos + 50
        tk.Label(well_chooser, text='Filename format: ({WWW} is the well name, e.g. "B02")').place(x=10,y=ypos)
        ypos = ypos + 30
        filename_format_var = tk.StringVar(value='{WWW}_fused_RGB.tif')
        entry = tk.Entry(well_chooser, textvariable=filename_format_var)
        entry.grid(row=1, sticky=tk.W)
        entry.place(x=10,y=ypos,width=400)
        
        ypos = ypos + 50
        B_continue = tk.Button(well_chooser, text='Continue', command=well_chooser.destroy, bg="green")
        B_continue.grid(row=2, sticky=tk.W, pady=4)
        B_continue.place(x=10,y=ypos)
        
        B_close = tk.Button(well_chooser, text='Close & quit', command=lambda:close_program(well_chooser,exit_code=True), bg="red")
        B_close.grid(row=2, sticky=tk.W, pady=4)
        B_close.place(x=110,y=ypos)
        well_chooser.mainloop()
        
        # Make a list of well names
        if allwells.get():  # the user wants to do all wells in the root directory
            well_list = next(os.walk(root.filename))[1]
        else:               # the user has specified the wells
            wells = wellstr.get()
            well_list = wells.split(',')
        
        # Make a list of filenames
        for well in well_list:
            
            foldername = get_file_name_from_format(foldername_format_var.get(), well)
            filename = get_file_name_from_format(filename_format_var.get(), well)
            root_path = os.path.join(root.filename, foldername)
            image_path = os.path.join(root_path, filename)
            # check if file exists
            if os.path.isfile(image_path):
                img_list.append((image_path, root_path)) # first entry is input image, second is output path
            else:
                raise ValueError('Sorry, the file '+image_path+' does not exist.')
                
    # MODEL TO USE ############################################################
    num_checked_boxes = np.sum([cyto_model.get(), nucleo_model.get(), self_trained.get()])
    if (num_checked_boxes != 1):
         raise ValueError('Please choose exactly one model.')
    
    model = None
    if cyto_model.get():
        model = 'cyto'
    elif nucleo_model.get():
        model = 'nuclei'
    elif self_trained.get():
        root = tk.Tk()
        root.filename = tk.filedialog.askopenfilename(initialdir=initial_dir, title="Select file with model parameters")
        model = root.filename
        root.destroy() # close GUI
        
    # Throw error if there are no images in fused_list, or if the output directory does not exist
    if len(img_list)==0:
        raise ValueError("Fatal error: there are no images to be processed!")
                
    return img_list,model,do_measure_var.get()

#%% 
def choose_image_with_gui(initial_dir, title):
    '''
    This function starts up a tkinter GUI which asks the user
    to open a tiff image.
    It outputs the path to the fused image (as a string).
    '''
    root = tk.Tk()
    root.filename =  tk.filedialog.askopenfilename(initialdir = initial_dir,title = title,filetypes = (('tif files','*.tif'),
                                                                                                                         ('png files','*.png')))
    path_to_fused_image = root.filename
    root.destroy() # close GUI
    
    return path_to_fused_image