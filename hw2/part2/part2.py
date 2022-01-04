# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:17:36 2021

@author: Ahmet Fatih Akcan
"""

import pickle
# import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import moviepy.editor as wr

def create_img(z): #the function creates and return image by given tensor
    img = gan(z, 0).numpy().squeeze()
    
    img = np.transpose(img, (1,2,0))
    img[img>1] = 1
    img[img<-1] = -1
    img = 255*(img+1) / 2
    
    return img



with open("stylegan3-t-ffhq-1024x1024.pkl", "rb") as f:
    a = pickle.load(f)
    
gan = a["G_ema"]

gan.eval() 

for param in gan.parameters():
    param.requires_grad = False
    

z_src = torch.randn(1, 512) #random tensor for source image
z_dst = torch.randn(1, 512) #random tensor for destination image

img_src = create_img(z_src) #creating the source image
img_dst = create_img(z_dst) #creating the destination image

cv2.imwrite("src.png", img_src[:,:,[2,1,0]]) # save the source image
cv2.imwrite("dst.png", img_dst[:,:,[2,1,0]]) # save the destination image
print("src, dst ok!") # to see the process while the slave named cpu is running for hours

img_list = [] 
img_list.append(img_src) # first frame is the source image

    
for i in range(1,100):
    img_now = create_img((z_src*(100-i)+z_dst*(i))/100) # creating combinations by ratio
    img_list.append(img_now) 
    print("i= ", i) # to see the process while the slave named cpu is running for hours
    
img_list.append(img_dst) # last frame is the destination image

clip = wr.ImageSequenceClip(img_list, fps=25) 
clip.write_videofile('morph.mp4', codec='libx264')  
