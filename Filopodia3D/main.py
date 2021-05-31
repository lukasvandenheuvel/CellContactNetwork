# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:09:08 2021

@author: lukasvandenheu
"""
import os
import torch
import torch.nn as nn
from torchsummary import summary
import time
import datetime
import matplotlib.pyplot as plt

from data import load_dataset, show_dataset
from model import try_gpu, Unet3D

from skimage import io

def list_to_tensor(list_of_tensors):
    N = len(list_of_tensors)
    [_,ch,h,w,z] = list_of_tensors[0].shape
    output_tensor = torch.zeros([N, ch, h, w, z], dtype = torch.float64)
    for i,tens in enumerate(list_of_tensors):
        output_tensor[i,:,:,:,:] = tens
    
    #if torch.cuda.is_available():
    #    output_tensor = output_tensor.cuda()
    return output_tensor

#%%
def fit_3D_unet(path,
                learning_rate, momentum, n_epoch,batch_size, 
                num_channels=1, test=False,
                input_dim = [128,128,32], div=1):
    
    
    
    # Load data
    field_size = input_dim[0]
    z_stack = input_dim[2]
    
    train_loader = load_dataset(path, field_size, overlap, z_stack, min_slice)
    image,label = list(zip(*train_loader))
    image = list_to_tensor(image)
    label = list_to_tensor(label).squeeze(1)
    num_load,num_channels,width,height,depth = image.size()
    
    N = int((1/div) * num_load) # number of images sent to GPU
    
    # Initialize 3D unet model
    model = Unet3D(num_channels=num_channels)
    
    # Send model to GPU (if a GPU is available)
    device = try_gpu()
    model.to(device)
    
    # Print overview of the model
    summary(model, tuple([num_channels] + input_dim))
    
    # Specifications of loss function    
    ws = torch.tensor([1,300], dtype = torch.float).cuda() # weights for each class
    loss_fun = nn.CrossEntropyLoss(weight=ws) # Loss function
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum) # 

    # shuffle dataset
    permutation = torch.randperm(num_load)
    image = image[permutation,:,:,:,:]
    label = label[permutation,:,:,:]
    
    # If test==True, split the dataset in a train and test set
    if test:
        split = round((3/4)*num_load)
        train_images = image[:split,:,:,:,:]
        train_labels = label[:split,:,:,:]
        test_images = image[split:,:,:,:,:]
        test_labels = label[split:,:,:,:]
    else:
        train_images = image
        train_labels = label

    losses = []
    accs = []
    
    for n in range(n_epoch):
        for i in range(div):
            
            # Make a subset of the training set and send it to the GPU
            subset_img = train_images[N*i:N*(i+1)]
            subset_label = train_labels[N*i:N*(i+1)]
            if torch.cuda.is_available():
                subset_img = subset_img.cuda()
                subset_label = subset_label.cuda()
            avg_loss, acc = model.train(subset_img, subset_label, batch_size, optimizer, loss_fun)
        
            # Copy average loss value to host memory and convert to numpy.
            # Use .detatch() to make sure the loss tensor does not require a gradient.
            if avg_loss.is_cuda:
                avg_loss = avg_loss.cpu().detach().numpy()
            else:
                avg_loss = avg_loss.detach().numpy()
            
            losses.append(avg_loss)
            accs.append(acc)
            
        print('Epoch {e} (of {n}): loss = {l:.2f}, Jaccard index = {a:.2f}%.'.format(e=n+1, n=n_epoch, l=avg_loss, a=acc))

    print('\nFinished Training')
    predictions = []
    if test:
        print('Testing...')
        predictions = model.test(test_images, test_labels, loss_fun)

    return model, losses, accs, predictions

#%%--------------------------------PARAMETERS---------------------------------

# Define train and target path
path = 'data/data_set'

# Training parameters
batch_size = 2
n_epoch = 500
learning_rate = 0.0001
momentum = 0.99
num_channels = 3
input_dim = [128,128,16]

# Define field dimensions
fov = 128 
overlap = 22 
z_stack = 16
min_slice = 10 
test = True
div = 2 # divide train set in 2 to save memory


# Empty the unused memory on GPU
torch.cuda.empty_cache()

# Do training
tic = time.perf_counter()
net, losses, accs, predictions = fit_3D_unet(path,
                                             learning_rate, momentum, n_epoch, batch_size,
                                             num_channels=num_channels,
                                             test=test, input_dim=input_dim, div=div)
toc = time.perf_counter()
#%%
# Save model parameters
date_and_time = str(datetime.datetime.now()).split()
date = date_and_time[0].replace('-', '_')
time = date_and_time[1].split('.')[0].replace(':','_')
output_path = os.path.join('./models', 'filo3D_d' + date + '_t' + time)
torch.save(net.state_dict(),output_path)

print('Total execution time: {:.2f} seconds'.format(toc- tic))

#%% Save test predictions
test_path = os.path.join(path, "test")
plt.close('all')
for i in range(len(predictions)):
    test_img,label,p = predictions[i]
    p = p[0,1,:,:,:].squeeze().permute(2,0,1).cpu().detach().numpy()
    img = test_img[0,:,:,:,:].squeeze().permute(3,1,2,0).cpu().detach().squeeze(0).numpy()
    label = label[0,:,:,:].squeeze().permute(2,0,1).cpu().detach().numpy()
    
    pred_out = os.path.join(test_path, str(i) + "_pred.tif")
    lbl_out = os.path.join(test_path, str(i) + "_label.tif")
    img_out = os.path.join(test_path, str(i) + "_img.tif")
    io.imsave(pred_out, p)
    io.imsave(lbl_out, label)
    io.imsave(img_out, img)
