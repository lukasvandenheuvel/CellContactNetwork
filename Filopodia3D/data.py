# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:11:43 2021

@author: lukasvandenheu
"""


# Setup
import random
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import elasticdeform.torch as etorch
import matplotlib.pyplot as plt
import skimage
import os

#%%
def edeform(input_tensor, target_tensor):
    """
    Applies random 3D elastic deformation & Random Rotation on the input and the target and 
    returns deformed input, output.

    input:  3D tensor of input image. With shape [x, y, z].
    target: 3D target tensor corresponding to the output Neural 
            network model. With shape [x, y, z]  
    """

    # Initialize random deformation
    displacement_val = np.random.randn(2, 3, 3) * 5
    displacement = torch.tensor(displacement_val)

    # Deform input and target --> ADD ROTATION
    input_deformed1 = etorch.deform_grid(input_tensor, displacement, order=3, axis=(0, 1))
    target_deformed1 = etorch.deform_grid(target_tensor, displacement, order=3, axis=(0, 1))

    # Random Rotation between [0-360] degree
    deg = random.uniform(0,360)
    rotate = transforms.Compose([transforms.RandomRotation((deg,deg))])

    input_deformed1 = rotate(input_deformed1)
    target_deformed1 = rotate(target_deformed1)

    return input_deformed1, target_deformed1

#%%
def load_dataset_old(input_path, target_path, num_load, num_channels, input_dim):

    '''
    Initialize loader for the training data and ground truth data
    List of data input containing tuples (input, label), input is 5D tensor, label is 4D tensor
    T = 'Train' or 'Test'
    device = 'cpu' or 'gpu'
    '''

    # intialise torch transformation to tensor
    transform_totens = transforms.Compose([transforms.ToTensor()])
    
    # initialize placeholders for output images and targets
    input_3D = torch.zeros([num_load, num_channels] + input_dim, dtype = torch.float64) #dtype int16 /
    target_3D = torch.zeros([num_load, 1] + input_dim, dtype = torch.float64)

    # loop through entire data folder
    for i in range(num_load): 
    
        input_img = Image.open(input_path + '/' + str(i) + '.tif')
        target_img = Image.open(target_path + '/' + str(i) + '.tif')
        
        width, height = input_img.size
        depth = input_img.n_frames

        # initialize placeholder tensors for input an target stacks
        in_t = torch.zeros([1, num_channels, height, width, depth], dtype = torch.float64)
        tar_t = torch.zeros([1, 1, height, width, depth], dtype = torch.float64)        

        for j in range(depth): 

            input_img.seek(j)                      
            target_img.seek(j)
            in_t[:, :, :, :, j] = transform_totens(input_img)
            tar_t[:, :, :, :, j] = transform_totens(target_img)
    
        # Interpolate in z-direction to get the desired depth
        in_interpolation = F.interpolate(in_t, size = input_dim)
        tar_interpolation = F.interpolate(tar_t, size = input_dim)

        # 0 = Background
        tar_interpolation = torch.where(tar_interpolation <= 30, 0, 0)
        # 1 = Cell body
        tar_interpolation = torch.where(torch.logical_and(tar_interpolation > 30, tar_interpolation < 70), 1, tar_interpolation.long())
        # 2 = Filopodia
        tar_interpolation = torch.where(tar_interpolation >= 70, 2, tar_interpolation.long())

        #train_loader.append((input_3D, target_3D))
        input_3D[i,:,:,:,:] = in_interpolation
        target_3D[i,:,:,:,:] = tar_interpolation
        
    if torch.cuda.is_available():
        input_3D = input_3D.cuda()
        target_3D = target_3D.cuda()

    return input_3D,target_3D.squeeze(1)

#%%

class DataLoader():
    def __init__(self, path, field_size, overlap, z_stack, min_slice):
        
        self.path = path
        self.field_size = field_size
        self.overlap = overlap
        self.z_stack = z_stack
        self.min_slice = min_slice

    #%%
    def normalize_image(self, img):
        return (img - np.mean(img)) / np.std(img)
        
    #%%
    def load_train(self):
      
      '''
        This function loads all the data in an img folder together with the target in the target folder into a trainloader.
        Every image will be reshaped, and later divided into multiple tiles with size "field_size" with "overlap" between field.
        Only images that are at least 1 field_size large with at least "min_slice" z-slices are loaded.
        Finally, the amount of z-slices is padded to "z_stack" and brought up to 2 * z_stack by interpolation.
      '''
      
      input_path = os.path.join(self.path, 'img')
      target_path = os.path.join(self.path, 'target')
      
      # Initialize d irectory
      ls = os.listdir(input_path)
      number_files = len(ls)
    
      # Initialize other parameters
      windows = []
      train_loader = []
      exc = 0
      count = 0
    
      # loop through entire data folder
      for i in range(number_files):
    
        img = Image.open(input_path + '/' + str(i) + '.tif')
        target = Image.open(target_path + '/' + str(i) + '.tif')
    
        # Get dimensions of the input
        width, height = img.size
        n_frames = img.n_frames
        depth = n_frames // 3 # divide by 3 because there are 3 channels
        
    
        if width >= self.field_size and height >= self.field_size and depth  >= self.min_slice: #only use images that are large enough
    
          print("Loading image %d"%count)
          count = count + 1
          
          windows = [width // (self.field_size - 0.5 * self.overlap), height // (self.field_size -  0.5 * self.overlap)] # this is the number of windows 
    
          w_out = int(windows[0] * self.field_size - (windows[0] - 1) * self.overlap)
          h_out = int(windows[1] * self.field_size - (windows[1] - 1) * self.overlap)
    
          # Define transforms
          transform_totens = transforms.Compose([transforms.ToTensor(),
                                                 transforms.CenterCrop((h_out, w_out))])
    
          # initialize placeholder tensors for input an target stacks
          in_t = torch.zeros([3, h_out, w_out, depth], dtype = torch.float64)
          tar_t = torch.zeros([h_out, w_out, depth], dtype = torch.float64)
          
          # tensors to store patches
          input_3D = torch.zeros([1, 3, self.field_size, self.field_size, self.z_stack], dtype = torch.float64)
          target_3D = torch.zeros([1, self.field_size, self.field_size, self.z_stack], dtype = torch.float64)
    
          for j in range(0,depth):
    
            # DAPI channel
            img.seek(3*j)
            input_np = self.normalize_image( np.array(img) )
            in_t[0, :, :, j] =  transform_totens(input_np)
    
            # GFP channel
            img.seek(3*j + 1)
            input_np = self.normalize_image( np.array(img) )
            in_t[1, :, :, j]  = transform_totens(input_np)
    
            # Actin channel
            img.seek(3*j + 2)
            input_np = self.normalize_image( np.array(img) )
            in_t[2, :, :, j]  = transform_totens(input_np)
            
            # target
            target.seek(j)
            tar_t[:, :, j] = transform_totens(target)
    
          w = [int(windows[0]), int(windows[1])]
          for k in range(w[0]):
            for l in range(w[1]):
                
              win = in_t[:, l * (self.field_size - self.overlap) : (l+1) * self.field_size - l * self.overlap, k * (self.field_size - self.overlap) : (k+1) * self.field_size - k * self.overlap, :]
              win = win.unsqueeze(0)  
    
              win_tar = tar_t[l * (self.field_size - self.overlap) : (l+1) * self.field_size - l * self.overlap, k * (self.field_size - self.overlap) : (k+1) * self.field_size - k * self.overlap, :]
              win_tar = win_tar.unsqueeze(0).unsqueeze(0) #first dimension is batch size and second dimenions is channels by convention
    
              # interpolate in z-direction to get the desired depth
              input_3D = F.interpolate(win, size = [self.field_size, self.field_size, self.z_stack])
              target_3D = F.interpolate(win_tar, size = [self.field_size, self.field_size, self.z_stack])
    
              # Append data, target pair in a trainloader
              train_loader.append((input_3D, target_3D)) 
    
        else:
            exc = exc + 1
    
      print(str(exc) + " of " + str(number_files) + " images have been excluded.") 
      print("Total number of patches loaded is %d."%len(train_loader))
      return train_loader

    #%% 
    def load_test(self):
        
        input_path = os.path.join(self.path, 'img')
          
        # Initialize directory
        ls = os.listdir(input_path)
        number_files = len(ls)
        
        # Initialize other parameters
        windows = []
        test_loader = []
        exc = 0
        count = 0
        
        # loop through entire data folder
        for i in range(number_files):
        
            img = Image.open(input_path + '/' + str(i) + '.tif')
            
            # Get dimensions of the input
            width, height = img.size
            n_frames = img.n_frames
            depth = n_frames // 3 # divide by 3 because there are 3 channels
            
            
            if width >= self.field_size and height >= self.field_size and depth  >= self.min_slice: #only use images that are large enough
            
                print("Loading image %d"%count)
                count = count + 1
                
                windows = [width // (self.field_size - 0.5 * self.overlap), height // (self.field_size -  0.5 * self.overlap)] # this is the number of windows 
              
                w_out = int(windows[0] * self.field_size - (windows[0] - 1) * self.overlap)
                h_out = int(windows[1] * self.field_size - (windows[1] - 1) * self.overlap)
              
                # Define transforms
                transform_totens = transforms.Compose([transforms.ToTensor(),
                                                   transforms.CenterCrop((h_out, w_out))])
              
                # initialize placeholder tensors for input an target stacks
                in_t = torch.zeros([3, h_out, w_out, depth], dtype = torch.float64)
                
                # tensors to store patches
                input_3D = torch.zeros([1, 3, self.field_size, self.field_size, self.z_stack], dtype = torch.float64)
                breakpoint()
                for j in range(0,depth):
                
                    # DAPI channel
                    img.seek(3*j)
                    input_np = self.normalize_image( np.array(img) )
                    in_t[0, :, :, j] =  transform_totens(input_np)
                    
                    # GFP channel
                    img.seek(3*j + 1)
                    input_np = self.normalize_image( np.array(img) )
                    in_t[1, :, :, j]  = transform_totens(input_np)
                    
                    # Actin channel
                    img.seek(3*j + 2)
                    input_np = self.normalize_image( np.array(img) )
                    in_t[2, :, :, j]  = transform_totens(input_np)
                    
                    w = [int(windows[0]), int(windows[1])]
                    for k in range(w[0]):
                        for l in range(w[1]):
                        
                          win = in_t[:, l * (self.field_size - self.overlap) : (l+1) * self.field_size - l * self.overlap, k * (self.field_size - self.overlap) : (k+1) * self.field_size - k * self.overlap, :]
                          win = win.unsqueeze(0)  
                          # interpolate in z-direction to get the desired depth
                          # input_3D = F.interpolate(win, size = [self.field_size, self.field_size, self.z_stack])                        
                          # Append data, target pair in a trainloader
                          input_3D = win
                          test_loader.append(input_3D) 
                
            else:
                exc = exc + 1
        
        print(str(exc) + " of " + str(number_files) + " images have been excluded.") 
        print("Total number of patches loaded is %d."%len(test_loader))
        return test_loader

    
    #%%
    def show_dataset(dataset, grayscale=True):
        """
        This function allows one to check if the data is loaded correctly. It will display three random samples form the set.
        """
        if grayscale:
            cmap = 'Greys_r'
        else:
            cmap = None
    
        plt.figure(figsize=(40, 3))
        for i in range(1, 6, 2):
            rand_idx = random.randrange(len(dataset))
            for j in range(0, 2):
                plt.subplot(1, 6, i + j)
    
                if j == 0:
                    _,_,_,_,z = dataset[rand_idx][j].size()
                    tensor = dataset[rand_idx][j][0,:,:,:,z//2].squeeze().permute(1,2,0)
                    plt.title('Input')           
    
                elif j == 1:
                    _,_,_,_,z = dataset[rand_idx][j].size()
                    tensor = dataset[rand_idx][j][0,0,:,:,z//2]
                    plt.title('Target')
        
                # If the img tensor is on the GPU, copy the tensor to host memory first
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                    
                plt.imshow(tensor.numpy(), aspect='equal')
                plt.subplots_adjust(top=1, bottom=0, left=0.05, right=0.95)
                plt.axis('off')
    
        plt.show()
        return True
