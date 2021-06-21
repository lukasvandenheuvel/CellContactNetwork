# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:12:05 2021

@author: lukasvandenheu
"""

import torch
import torch.nn as nn
import random

#%%
def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

#%%
# Double convolution
def double_conv(in_c, step_c, out_c):
    conv = nn.Sequential(
          nn.Conv3d(in_c, step_c, kernel_size=3, padding=1, padding_mode='zeros'),
          nn.BatchNorm3d(step_c),
          nn.ReLU(),
          nn.Conv3d(step_c, out_c, kernel_size=3, padding=1, padding_mode='zeros'),
          nn.BatchNorm3d(out_c),
          nn.ReLU()
        )
    return conv

#%%
# Crop input for concatenation
def crop_img(tensor, target_tensor):

    target_size_xy = target_tensor.size()[2]
    tensor_size_xy = tensor.size()[2]
    delta_xy = tensor_size_xy - target_size_xy
    delta_xy = delta_xy // 2

    target_size_z = target_tensor.size()[4]
    tensor_size_z = tensor.size()[4]
    delta_z = tensor_size_z - target_size_z
    delta_z = delta_z // 2

    return tensor[:, :, delta_xy:tensor_size_xy-delta_xy, delta_xy:tensor_size_xy-delta_xy, delta_z:tensor_size_z-delta_z]

#%%
def calculate_jaccard_index(prediction, label):
    '''
    Calculates the Jaccard index of prediction and label.
    J = intersection(A,B) / union(A,B)
    '''
    
    intersection = (prediction * label).sum()
    union = ((prediction + label) > 0).sum()
    # If the union is 0, the Jaccard index is 100%. 
    # The image is correctly classified as all background.
    if union==0 and intersection==0:
        J = 1
    else:
        J = intersection / union
    return J
    

#%%
# 3D unet architecture
class Unet3D(nn.Module):
    ''' 3DUnet; input [1, 1, 132, 132, 124] ; return [1, 44, 44, 36]'''
    def __init__(self, num_channels):
        super(Unet3D, self).__init__()

        self.num_channels = num_channels
        self.max_pool_2x2x2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.down_conv1 = double_conv(self.num_channels, 32, 64)
        self.down_conv2 = double_conv(64, 64, 128)
        self.down_conv3 = double_conv(128, 128, 256)
        self.down_conv4 = double_conv(256, 256, 512)

        self.up_trans1 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.up_trans2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.up_trans3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)

        self.up_conv1 = double_conv(768, 256, 256)
        self.up_conv2 = double_conv(384, 128, 128)
        self.up_conv3 = double_conv(192, 64, 64)

        self.final_conv = nn.Conv3d(64, 2, kernel_size=1)
        self.final_layer = nn.Sigmoid()

    def forward(self, x):
        
        # Encoder
        x1 = self.down_conv1(x)
        x = self.max_pool_2x2x2(x1)
        x2 = self.down_conv2(x)
        x = self.max_pool_2x2x2(x2)
        x3 = self.down_conv3(x)
        x = self.max_pool_2x2x2(x3)
        x = self.down_conv4(x)

        # Decoder
        x = self.up_trans1(x)
        y1 = crop_img(x3, x)
        x = self.up_conv1(torch.cat([x, y1], 1))

        x = self.up_trans2(x)
        y2 = crop_img(x2, x)
        x = self.up_conv2(torch.cat([x, y2], 1))

        x = self.up_trans3(x)
        y3 = crop_img(x1, x)
        x = self.up_conv3(torch.cat([x, y3], 1))
        x = self.final_conv(x)
        x = self.final_layer(x)

        return x

    def train(self, train_images, train_labels, batch_size, optimizer, criterion):
        """
        Trains network for one epoch in batches.
        Args:
          train_loader: Data loader for training set.
          model: Neural network model.
          optimizer: Optimizer (e.g. SGD).
          criterion: Loss function (e.g. cross-entropy loss).
        """
        avg_loss = 0
        correct = 0
        num_iterations = 0 

        n_img = train_images.size()[0]
        permutation = torch.randperm(n_img)
        
        # iterate through batches
        for i in range(0, n_img, batch_size):
            
            indices = permutation[i:i+batch_size]
            input_img = train_images[indices,:,:,:,:]
            label = train_labels[indices,:,:,:]
            
            # Set the gradients of all model parameters to 0.
            # If we don't do this, Pytorch accumulates the gradients on subsequent backward passes.
            optimizer.zero_grad()
            
            # forward + backward + optimize
            output = self.forward(input_img.float())      
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

            # keep track of loss and accuracy
            avg_loss += loss
            num_iterations += 1
            predicted_filo = (output[:,1,:,:,:] > 0.5) # binarize the second channel (filo)
            correct += calculate_jaccard_index(predicted_filo, label)
            
        acc = 100 * correct / (num_iterations)            

        return avg_loss/num_iterations, (acc)


    def test(self, test_images, test_labels, criterion ):

        with torch.no_grad(): # omitting the gradient saves a lot of memory
            avg_loss = 0
            correct = 0
            num_imgs = test_images.size()[0]
            predictions = []
    
            for i in range(num_imgs):
                input_img = test_images[i,:,:,:,:].unsqueeze(0)
                label = test_labels[i,:,:,:].unsqueeze(0)
                if torch.cuda.is_available():
                    input_img = input_img.cuda()
                    label = label.cuda()
                
                # Do a forward pass
                output = self.forward(input_img.float())
                loss = criterion(output, label.long())
    
                # keep track of loss and accuracy
                avg_loss += loss
                _, predicted = torch.max(output.data, 1)
                
                predicted_filo = (output[:,1,:,:,:] > 0.5) # binarize the second channel (filo)
                correct +=  calculate_jaccard_index(predicted_filo, label)
                predictions.append((input_img.cpu(), label, output.cpu()))
                
        acc = 100*correct / num_imgs

        print('final avg loss: ' + str(avg_loss / num_imgs))
        print('final avg Jaccard index: ' + str(acc) + '%')
        return predictions, avg_loss/num_imgs, (acc)
        
