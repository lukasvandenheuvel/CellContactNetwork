#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:07:00 2021

@author: lukasvandenheuvel
"""

import numpy as np
import pandas as pd
from torchvision import transforms
from skimage import io
from PIL import Image, ImageOps
import random
import copy
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
from skimage.filters import gaussian
import os
from cellpose import models
use_GPU = models.use_gpu()


#%%
def transformer(img, label):
    '''
    This function does data augmentation on an image and its corresponding label.
    The operations done are (in chronological order):
        - Horizontal flip (with a probability of 50%)
        - Vertical flip (with a probability of 50%)
        - Affine transform: rotation, translation, cropping, scaling and shear.
        - Gaussian blur on img only (with a probability of 50%)
        - Add Gaussian noise on img only (with a probability of 50%).
    
    Parameters
    ----------
    img : PIL image (RGB)
        Input RGB image.
    label : PIL image (grayscale)
        Corresponding label.

    Returns
    -------
    transformed_img : numpy array
        Transformed RGB image with the same dimensions as img.
    transformed_label : numpy array
        Transformed label with the same dimensions as label.

    '''
    
    # image and mask are PIL image object. 
    img_w, img_h = img.size
    num_channels = len(img.getbands())
    image = copy.deepcopy(img)
    
    # Random horizontal flipping
    if random.random() < 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)

    # Random vertical flipping
    if random.random() < 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)
  
    # Random affine
    affine_param = transforms.RandomAffine.get_params(
                                                degrees = [-180, 180], 
                                                translate = [0.3,0.3],  
                                                img_size = [img_w, img_h], 
                                                scale_ranges = [1, 1.3], 
                                                shears = [2,2])
    image = TF.affine(image, 
                      affine_param[0], affine_param[1],
                      affine_param[2], affine_param[3])
    label = TF.affine(label, 
                     affine_param[0], affine_param[1],
                     affine_param[2], affine_param[3])

    image = np.array(image)
    label = np.array(label)
    image_type = image.dtype

    # Random GaussianBlur -- only for images
    if random.random() < 0.5:
        sigma_param = random.uniform(0.01, 1.5)
        image = 255*gaussian(image, sigma=sigma_param)
    
    # Random Gaussian Noise -- only for images
    if random.random() < 0.5:
        factor_param = random.uniform(0.01,0.3)
        image = image + np.asarray( np.abs( factor_param * image.std() * np.random.randn(image.shape[0], image.shape[1], num_channels) ), dtype=image_type)

        
    return np.asarray(image, dtype=image_type), label

#%%
def load_train_data(train_path, label_path):
    '''
    Takes as input a path to data folder as string.
    Training images should be TIF, labels should be PNG. 
    Filenames should be '001.tif', '002.tif', ... for images 
    and '001.png', '002.png', ... for labels.
    '''
    
    # find nr of train images
    nr_train_images = 0
    file_list = os.listdir(train_path)
    for file in file_list:
        if ('.tif' in file or '.png' in file):
            nr = int(file.split('.')[0])
            if nr > nr_train_images:
                nr_train_images = nr
    
    train_set = []
    for i in range(1,nr_train_images+1):
        img_file = os.path.join(train_path, "{:03n}".format(i) + '.tif')
        label_file = os.path.join(label_path, "{:03n}".format(i) + '.png')
        img = Image.open(img_file)
        label = ImageOps.grayscale( Image.open(label_file) )
        train_set.append((img,label))
    
    return train_set

#%%
def augment_data(train_set, nr_augmentations_per_image):
    '''
    This function does data augmentation on every (img,label) pair in train_set.
    For each image, it does nr_augmentations_per_image transformations and it 
    appends these transformations to the train set.
    '''
    augmented_data = []
    train_set_numpy = []
    for img,mask in train_set:
        # Convert original data to numpy
        train_set_numpy.append((np.array(img) / 255, np.array(mask)))
        for i in range(nr_augmentations_per_image):
            tf_img, tf_mask = transformer(img, mask)
            augmented_data.append((tf_img / 255, tf_mask))
        
    return train_set_numpy + augmented_data

#%% 
def pil_to_numpy(list_of_pil):
    '''
    Convert pil images to numpy arrays
    '''
    list_of_numpy = []
    for img,mask in list_of_pil:
        list_of_numpy.append((np.array(img) / 255, np.array(mask)))
    return list_of_numpy
            
#%%
def load_test_data(path_to_data):
    '''
    Takes as input a path to data folder as string.
    It outputs a list of images (.tif or .png) in the data folder together with
    a list of filenames.
    '''
    file_list = os.listdir(path_to_data)
   
    test_data = []
    test_data_filenames = []
    for file in sorted(file_list):
        if (not('_predict' in file) and ('.tif' in file or '.png' in file)):
            test_data.append(io.imread(os.path.join(path_to_data,file)))
            test_data_filenames.append(file)
    
    return test_data,test_data_filenames

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
        Cellpose pre-trained nucleus model, or any other name for a custom-trained model. 
        
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
def save_masks(output_path, mask_list, flow_list, test_data_filenames):
    '''
    Save masks and flow predictions made by cellpose in output_path.
    '''
    for mask, flow, path_to_test_img in zip(mask_list, flow_list, test_data_filenames):
        nr = path_to_test_img.split('.')[0]
        io.imsave( os.path.join(output_path, nr+'_predict.tif'), mask )
        
        for f_nr,f in enumerate(flow):
            io.imsave( os.path.join(output_path, nr+'_predict_flow'+str(f_nr)+'.tif'), f )
            
    return True
            
#%% ------------------------------ START CODE ---------------------------------

# Specify training parameters -------------------------------------------------
learning_rate = 0.05
momentum = 0.9
channel_to_segment = 'R' # choose 'R', 'G', or 'B'.
nucleus_channel = 'B'    # choose 'R', 'G', or 'B' or 'None'
batch_size = 4
n_epochs = 2
weight_decay = 0.00001
num_augmentations_per_image = 2

# Specifications of the pretrained model --------------------------------------
path_to_pretrained_model = None # Path to the pretrained model file (example: './models/cellpose_residual_on_style_on_concatenation_off_Cellpose_2021_05_04.236206'). Set to None if you want to train from scratch.

# Path to images and annotations ----------------------------------------------
train_path = './data/train/ais/image'
label_path = './data/train/ais/label'

# Path to specialized dataset -------------------------------------------------
specialised_train_path = './data/train/neuroblastoma/image' # Set to None if you don't want to use a specialized dataset
specialised_label_path = './data/train/neuroblastoma/label' # Set to None if you don't want to use a specialized dataset

# Path to test images ---------------------------------------------------------
test_path = './data/test/image'              # Set to None if you don't have test images

# Output path -----------------------------------------------------------------
prediction_path = './data/test/image'        # where the predicted test images will be saved (only used if test_path is not None)
save_path = './'                             # where the model parameters will be saved

#%% Load model with pretrained weights ---------------------------------------
model = models.CellposeModel(gpu=use_GPU, pretrained_model=path_to_pretrained_model)

#%% Load train and test dataset -----------------------------------------------
print('>>>> Loading and augmenting data...')
pre_train_set = load_train_data(train_path,label_path)
aug_train_set = augment_data(pre_train_set, num_augmentations_per_image)

spec_train_set = [] 
if not(specialised_train_path==None):
    print('>>>> Loading specialized dataset...')
    spec_train_set = load_train_data(specialised_train_path,specialised_label_path)
    spec_train_set = pil_to_numpy(spec_train_set)
    
train_set = aug_train_set + spec_train_set # combine augmented dataset and specialized dataset
random.shuffle(train_set)
train_images,train_labels = list(zip(*train_set))

num_training_images = len(train_images)
print(">>>> Loaded {} training images".format(num_training_images))
test_data,test_data_filenames = load_test_data(test_path)

#%% Show 5 train images ------------------------------------------------------

f, ax = plt.subplots(2, 5, figsize=(12, 8))
for i in range(5):
    nr = random.randint(0,num_training_images-1)
    img = train_images[nr]
    label = train_labels[nr]
    ax[0,i].imshow(img)
    ax[1,i].imshow(label)
f.show()

#%% Train model ---------------------------------------------------------------
channels = find_channels('custom', channel_to_segment, nucleus_channel)
model_path = model.train(list(train_images), list(train_labels), test_data=None, channels=channels, save_path=save_path, learning_rate=learning_rate, n_epochs=n_epochs,  momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

#%% Save parameters file ------------------------------------------------------
model_name = model_path.split('/')[-1]
model_name_without_extension = model_name.split('.')[0]
params_csv_file = model_name_without_extension +  '.csv'
path_to_params = os.path.join( '/'.join(model_path.split('/')[0:-1]), params_csv_file)

params = [num_training_images, learning_rate, momentum, batch_size, channels, n_epochs, weight_decay]
param_names = ['num_training_images', 'learning_rate', 'momentum', 'batch_size', 'channels', 'n_epochs', 'weight_decay']
pd_params = pd.DataFrame([params], columns=param_names)
save = pd_params.to_csv(path_to_params, index=None)

#%% Evaluate model -----------------------------------------------------------
if not(test_path==None):
    mask_list, flow_list, styles = model.eval(test_data, channels = [1,3])
    saved = save_masks(prediction_path, mask_list, flow_list, test_data_filenames)
    print('>>>> Predictions saved successfully in ' + prediction_path)