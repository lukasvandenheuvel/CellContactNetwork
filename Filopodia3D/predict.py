# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:13:50 2021

@author: lukasvandenheu
"""
from model import Unet3D
from data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

def find_max_coordinates(train_loader):
    
    # init dicts with zeros
    k_max_dict = {}
    l_max_dict = {}
    i_array = []
    for _,coordinates in train_loader:
         i,k,l = coordinates
         k_max_dict[i] = 0
         l_max_dict[i] = 0
         i_array.append(i)
        
    for _,coordinates in train_loader:
        i,k,l = coordinates
        if k > k_max_dict[i]:
            k_max_dict[i] = k
        if l > l_max_dict[i]:
            l_max_dict[i] = l
            
    return k_max_dict,l_max_dict,max(i_array)+1

#%%

num_channels = 3
input_dim = [128,128,16]

# Define field dimensions
fov = 128 
overlap = 22 
z_stack = 16
min_slice = 10 
test = True
div = 2 # divide train set in 2 to save memory

# where are the test images?
path_to_test = r'M:\tnw\bn\dm\Shared\Lukas\NetworkAnalysis\CellContactNetwork\Filopodia3D\data\test_images_Unet\JJ017'
path_to_model = r'M:\tnw\bn\dm\Shared\Lukas\NetworkAnalysis\CellContactNetwork\Filopodia3D\models\filo3D_d2021_06_15_t16_15_45'

print("Loading test...")
data = DataLoader(path_to_test, fov, overlap, z_stack, min_slice)
test_loader = data.load_test()

print(test_loader[0][0].shape)

# Load model
model = Unet3D(num_channels)
model.load_state_dict(torch.load(path_to_model))



#%% Make prediction
# Empty the unused memory on GPU
torch.cuda.empty_cache()
predictions = []
for test_img,_ in test_loader:
    predictions.append(model.forward(test_img.float()))

#%% plot result
nr = 0
nr = nr+1
predict = predictions[nr]
test_img = test_loader[nr][0]
plt.close('all')
predict_show = predict[0,1,:,:,:].squeeze().permute(2,0,1).cpu().detach().numpy()
test_img_show = test_img[0,:,:,:,:].squeeze().permute(3,1,2,0).cpu().detach().squeeze(0).numpy()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(test_img_show[8,:,:,:])

plt.subplot(1,2,2)
plt.imshow(predict_show[8,:,:])

print("Done!!!!!")

#%% Stitch tiles back to input image
k_max_dict,l_max_dict,num_images = find_max_coordinates(test_loader)
#%%
count = 0
output_images = []
output_predictions = []

for t in range(num_images):
    
    
    k_max = k_max_dict[t]
    l_max = l_max_dict[t]
    
    pred = torch.zeros(1,2,k_max*fov-(k_max-1)*overlap,l_max*fov-(l_max-1)*overlap,z_stack)
    img = torch.zeros(1,3,k_max*fov-(k_max-1)*overlap,l_max*fov-(l_max-1)*overlap,z_stack)
    
    for kk in range(k_max+1):
        for ll in range(l_max+1):
            
            x_cor = kk * fov - (kk - 1) * fov
            y_cor = ll * fov - (ll - 1) * fov
            
            pred[:,:, x_cor:x_cor+fov, y_cor:y_cor+fov, :] = predictions[count]
            img[:,:, x_cor:x_cor+fov, y_cor:y_cor+fov, :] = test_loader[count][0]
            
            count = count + 1

    output_predictions.append(pred)
    output_images.append(img)
  

#%%
pred_to_plot = output_predictions[0]
img_to_plot = output_images[0]
predict_show = pred_to_plot[0,1,:,:,:].squeeze().permute(2,0,1).cpu().detach().numpy()
img_show = img_to_plot[0,:,:,:,:].squeeze().permute(3,1,2,0).cpu().detach().squeeze(0).numpy()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_show[2,:,:,:])

plt.subplot(1,2,2)
plt.imshow(predict_show[2,:,:])
          