# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:57:37 2021

@author: lukasvandenheu
"""

import numpy as np             # for scientific computing
import pandas as pd            # for reading and writing csv files
from skimage import io         # for image processing
import os                      # operating system-specific
from cellpose import models    
import matplotlib.pyplot as plt
use_GPU = models.use_gpu()

#%%
#%%
def load_data(path_to_data):
    '''
    Takes as input a path to data folder as string.
    It outputs a list of images (.tif or .png) in the data folder
    '''
    file_list = os.listdir(path_to_data)
    image_filenames = []
    for file in sorted(file_list):
        if ('.tif' in file or '.png' in file):
            image_filenames.append(file)

    return [io.imread(os.path.join(path_to_data,f)) for f in image_filenames], image_filenames


def find_all_model_files(path_to_models):
    '''
    Takes as input a path to model folder as string.
    It outputs a list filenames of all cellpose models in the model directory.
    '''
    file_list = os.listdir(path_to_models)
    model_names = []
    for file in file_list:
        if 'cellpose' in file and not('.csv' in file):
            model_names.append(file)
    return model_names


def calculate_jaccard_index(A, B):
    '''
    Calculates the Jaccard index between two binary images A and B.
    J = (intersection(A,B) / union(A,B))
    If A and B are equal, J=1 and if they are complementary J=0.
    '''
    intersection = A*B
    union = (A+B > 0)
    return np.sum(intersection) / np.sum(union)

def evaluate_cellpose_model(path_to_model, test_data, test_labels, channels=[1,3], flow_threshold=0.4):
    '''
    This function compares human-annotated test_labels with outputs of the
    cellpose prediction for a set of test images.
    For each prediction-label pair, it calculates the Jaccard index, and it 
    outputs them all as a list.
    
    Inputs
    ------
        path_to_model (string)
        Path to model weights.
        
        test_data (list of numpy arrays)
        Test images to predict (RGB).
        
        test_labels (list of numpy arrays)
        Human-annotated labels which correspond to test_data.
        
        channels (list of 2 integers, optional, default=[1,3])
        Channels to predict. First channel is cyto, second is nucleus.
        
        flow_threshold (float, optional, default=0.4)
        Is used for gradient backtracing. Set lower to segment less cells and
        higher to segment more cells.
        
    OUTPUT
    ------
        J (list of float)
        Jaccard index of each prediction-label pair.
    '''
    # Load model with pretrained weights
    model = models.CellposeModel(gpu=use_GPU, pretrained_model=path_to_pretrained_model)
    # Evaluate test images
    mask_list, flow_list, styles = model.eval(test_data, channels=channels, flow_threshold=flow_threshold)
    # Calculate Jaccard index
    J = np.zeros(len(mask_list))
    c = 0
    for prediction,label in zip(mask_list, test_labels):
        binary_pred = prediction>0
        binary_label = label>0
        J[c] = calculate_jaccard_index(binary_pred, binary_label)
        c = c+1
    return J,mask_list,flow_list

def read_parameters_file(model_name, path_to_models):
    '''
    Read csv file in which the training parameters are stored.
    Outputs a Pandas dataframe.
    '''
    model_name_without_extension = model_name.split('.')[0]
    params_csv_file = model_name_without_extension +  '.csv'
    path_to_params = os.path.join( path_to_models, params_csv_file )
    return pd.read_csv(path_to_params)

def save_masks(output_path, mask_list, flow_list, test_data_filenames):
    for mask, flow, path_to_test_img in zip(mask_list, flow_list, test_data_filenames):
        nr = path_to_test_img.split('.')[0]
        io.imsave( os.path.join(output_path, nr+'_predict.tif'), mask )
        
        for f_nr,f in enumerate(flow):
            io.imsave( os.path.join(output_path, nr+'_predict_flow'+str(f_nr)+'.tif'), f )
            
    return True

#%% ------------------------------- Set parameters ----------------------------
flow_threshold = 0.4

path_to_models = './models/AISModel'
models_to_test = 'all' # set to 'all' to test all models in the directory path_to_models, or make a list with the model filenames you want to test.

save_model = -1 # save predictions of last model in the list
test_path = './data/test' # where the folders 'image' and 'label' are with test images and labels
prediction_path = './data/test/predictions' # where you want the predictions to be stored

## Load test data -------------------------------------------------------------
test_data,test_data_filenames = load_data(os.path.join(test_path, 'image'))
test_labels,test_label_filenames = load_data(os.path.join(test_path, 'label'))

if (models_to_test=='all'):
    models_to_test = find_all_model_files(path_to_models)

params_df_list = []
xlabels = []
num_models = len(models_to_test)
mean_J = np.zeros((num_models))
std_J  = np.zeros((num_models))
model_nrs = np.zeros((num_models))

for i,model_name in enumerate(models_to_test):
    
    # Read parameters
    params_df = read_parameters_file(model_name, path_to_models)
    xlabels.append(str(params_df.iloc[0]).split('Name')[0])
    params_df_list.append(params_df)
    
    # Evaluate Jaccard index
    path_to_pretrained_model = os.path.join(path_to_models, model_name)
    J,mask_list,flow_list = evaluate_cellpose_model(path_to_pretrained_model, test_data, test_labels, channels=[1,3], flow_threshold=flow_threshold)
    mean_J[i] = np.mean(J)
    std_J[i] = np.std(J)
    model_nrs[i] = i
    
    if (i == save_model-1):
        saved = save_masks(prediction_path, mask_list, flow_list, test_data_filenames)

#%% Build the plot
fig, ax = plt.subplots()
ax.bar(model_nrs, mean_J, yerr=std_J, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Mean Jaccard index')
ax.set_xticks(model_nrs)
ax.set_xticklabels(xlabels)
ax.set_title('Quality of predictions')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()

