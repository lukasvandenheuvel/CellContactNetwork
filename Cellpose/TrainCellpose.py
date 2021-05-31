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
def save_masks(output_path, mask_list, flow_list, test_data_filenames):
    for mask, flow, path_to_test_img in zip(mask_list, flow_list, test_data_filenames):
        nr = path_to_test_img.split('.')[0]
        io.imsave( os.path.join(output_path, nr+'_predict.tif'), mask )
        
        for f_nr,f in enumerate(flow):
            io.imsave( os.path.join(output_path, nr+'_predict_flow'+str(f_nr)+'.tif'), f )
            
    return True
            
#%% Init
learning_rate = 0.05
momentum = 0.9
channels = [1,3]
batch_size = 4
n_epochs = 7000
weight_decay = 0.00001
num_augmentations_per_image = 2

path_to_models = r'M:\tnw\bn\dm\Shared\Lukas\NetworkAnalysis\CellContactNetwork\Cellpose\models'
pretrained_model_name = 'cellpose_residual_on_style_on_concatenation_off_models_2021_04_21_08_59_47.100849'

train_path = r'M:\tnw\bn\dm\Shared\Jurriaan\Trainingdata\train\ais\image'
label_path = r'M:\tnw\bn\dm\Shared\Jurriaan\Trainingdata\train\ais\label'
specialised_train_path = r'M:\tnw\bn\dm\Shared\Jurriaan\Trainingdata\train\glioma\image'
specialised_label_path = r'M:\tnw\bn\dm\Shared\Jurriaan\Trainingdata\train\glioma\label'
test_path = r'M:\tnw\bn\dm\Shared\Jurriaan\Trainingdata\test\image'
prediction_path = r'M:\tnw\bn\dm\Shared\Jurriaan\Trainingdata\test\predictions'
save_path = r'M:\tnw\bn\dm\Shared\Lukas\NetworkAnalysis\CellContactNetwork\Cellpose\models'

#%% Load model with pretrained weights
path_to_pretrained_model = os.path.join(path_to_models, pretrained_model_name)
path_to_pretrained_model = None
model = models.CellposeModel(gpu=use_GPU, pretrained_model=path_to_pretrained_model)

#%% Load train and test dataset 
pre_train_set = load_train_data(train_path,label_path)
aug_train_set = augment_data(pre_train_set, num_augmentations_per_image)
spec_train_set = load_train_data(specialised_train_path,specialised_label_path)
spec_train_set = pil_to_numpy(spec_train_set)

train_set = aug_train_set + spec_train_set
random.shuffle(train_set)
train_images,train_labels = list(zip(*train_set))

num_training_images = len(train_images)
print("Loaded {} training images".format(num_training_images))
test_data,test_data_filenames = load_test_data(test_path)

#%% Show 5 images

f, ax = plt.subplots(2, 5, figsize=(12, 8))
for i in range(5):
    nr = random.randint(0,num_training_images-1)
    img = train_images[nr]
    label = train_labels[nr]
    ax[0,i].imshow(img)
    ax[1,i].imshow(label)


#%% Train model
model_path = model.train(list(train_images), list(train_labels), test_data=None, channels=channels, save_path=save_path, learning_rate=learning_rate, n_epochs=n_epochs,  momentum=momentum, weight_decay=weight_decay, batch_size=batch_size)

#%% Save parameters file
model_name = model_path.split('/')[1]
model_name_without_extension = model_name.split('.')[0]
params_csv_file = model_name_without_extension +  '.csv'
path_to_params = os.path.join( model_path.split('/')[0], params_csv_file)

params = [num_training_images, learning_rate, momentum, batch_size, channels, n_epochs, weight_decay]
param_names = ['num_training_images', 'learning_rate', 'momentum', 'batch_size', 'channels', 'n_epochs', 'weight_decay']
pd_params = pd.DataFrame([params], columns=param_names)
save = pd_params.to_csv(path_to_params, index=None)

#%% Evaluate model
mask_list, flow_list, styles = model.eval(test_data, channels = [1,3])
saved = save_masks(prediction_path, mask_list, flow_list, test_data_filenames)