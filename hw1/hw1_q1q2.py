#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:04:02 2021

Author: Ahmet Fatih Akcan
Student Number: 150210707

Notes: The script of both part1 and part2 for HW1
"""
import cv2 as cv
import os
import numpy as np
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import scipy.stats


class HW1_q1q2():
    def __init__(self):
# =============================================================================
#         reads background and resizes it, also reads cat images
# =============================================================================
        self.background = cv.imread('hw1_material/Malibu.jpg')
        
        self.bg_h = self.background.shape[0]
        self.bg_w = self.background.shape[1]
        self.ratio = 360/self.bg_h
        
        self.background = cv.resize(self.background, (int(self.bg_w * self.ratio), 360))
        print(self.background.shape)
        
        self.cat_list = list()
        for i in range(180):
            now = "hw1_material/cat/cat_" + str(i) + ".png"
            self.cat_list.append(cv.imread(now))

    
    def cdf(self, img):
# =============================================================================
#         my custom cdf function instead of plt.hist()  (to speed up the process)
# =============================================================================
        hist, bins = np.histogram(img.flatten(), 256, density=True)
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        return cdf
    
    
    def LUT(self, cdf_I, cdf_J):
        """
            creating Look Up Table for Histogram Matching
        """
        LUT=np.zeros((256,1))
        gj=0
        for gi in range(256):
            while ((cdf_J[gj] < cdf_I[gi]) and gj<256-1):
                gj = gj+1
            LUT[gi]=gj
        return LUT
    
    
    def flip_horizontally(self, img):
# =============================================================================
#             custom function to flip an image horizontally  
# =============================================================================
        height, width = img.shape[:2]
        tform = np.float32([ [-1, 0, width-1], [0, 1, 0]])
        flipped = np.zeros([height, width], dtype=np.uint8)
        flipped = cv.warpAffine(img, tform, (width,height))
        return flipped
    
    
    def shift_to_rightSide(self, img):
# =============================================================================
#             custom function to shift an image to right side of the frame
# =============================================================================
        height, width = img.shape[:2]
        tform = np.float32([ [1, 0, int(self.bg_w * self.ratio)-width], [0, 1, 0]])
        shifted = np.zeros([height, width], dtype=np.uint8)
        shifted = cv.warpAffine(img, tform, (width,height))
        return shifted
        
    
    def q1(self):
# =============================================================================
#         codes for part1
# =============================================================================
        
        img_list = list()
        for cat_now in self.cat_list:
            empty = np.zeros((360,926,3), dtype='uint8') #creating an empty frame
            empty[:,:,0] = 16 #R
            empty[:,:,1] = 235 #G
            empty[:,:,2] = 15 #B
            empty[:cat_now.shape[0], :cat_now.shape[1], :] = cat_now #adding cat images to resized green new frame
            cat_inv = self.shift_to_rightSide(self.flip_horizontally(empty)) #creating inverse cat's (flip and shift)

            #masking cat images and overlay them to the background image (cats onto malibu :d)
            foreground = np.logical_or(cat_now[:,:,1] < 180, cat_now[:,:,0] > 150)
            nonzero_x, nonzero_y = np.nonzero(foreground)
            nonzero_cat_values = cat_now[nonzero_x, nonzero_y, :]
            new_frame = self.background.copy()
            new_frame[nonzero_x, nonzero_y] = nonzero_cat_values       
            
            #masking inverse cat images and overlay them to the recent background image (inverse cats onto malibu&cats :d)
            foreground = np.logical_or(cat_inv[:,:,1] < 180, cat_inv[:,:,0] > 150)
            nonzero_x, nonzero_y = np.nonzero(foreground)
            nonzero_cat_values = cat_inv[nonzero_x, nonzero_y, :]
            new_frame[nonzero_x, nonzero_y] = nonzero_cat_values 
            
            
            new_frame = new_frame[:,:,[2,1,0]] #convert result from RGB to BGR
            img_list.append(new_frame) # add the last frame to the image list which will be used to create video
            
        #creating clip and the add audio
        clip = mpy.ImageSequenceClip(img_list, fps=25)
        audio = mpy.AudioFileClip('hw1_material/selfcontrol_part.wav').set_duration(clip.duration)
        clip = clip.set_audio(audioclip=audio)
        clip.write_videofile('part1_video.mp4', codec='libx264')
        print("*****************PART1 COMPLETED!!!*****************")

    
    def q2(self):
# =============================================================================
#         codes for part2
# =============================================================================
        
        img_list = list()
        count=0 #to watch the processing sequence
        for cat_now in self.cat_list:
            empty = np.zeros((360,926,3), dtype='uint8') #creating an empty frame
            empty[:,:,0] = 16 #R
            empty[:,:,1] = 235 #G
            empty[:,:,2] = 15 #B
            empty[:cat_now.shape[0], :cat_now.shape[1], :] = cat_now #adding cat images to resized green new frame
            cat_inv = self.shift_to_rightSide(self.flip_horizontally(empty)) #creating inverse cat's (flip and shift)
            #---------------------------------------------------
            
            #------------------------single cat-----------------
            foreground = np.logical_or(cat_now[:,:,1] < 180, cat_now[:,:,0] > 150)
            nonzero_x, nonzero_y = np.nonzero(foreground)
            nonzero_cat_values = cat_now[nonzero_x, nonzero_y, :]
            new_frame_single = self.background.copy()
            new_frame_single[nonzero_x, nonzero_y] = nonzero_cat_values
            #-------------------------------------------------------
            
            #------------cdf's of target now---------------
            h,w = new_frame_single.shape[:2]
            
            r=new_frame_single[:,:,0]
            g=new_frame_single[:,:,1]
            b=new_frame_single[:,:,2]
                        
            r_flattened = r.reshape([h*w])
            g_flattened = g.reshape([h*w])
            b_flattened = b.reshape([h*w])
            
            
            cdf_r_t = self.cdf(r_flattened)
#            cdf_r_t = plt.hist(r_flattened, bins=256, normed=True, cumulative=True) 
#            plt.show()
            
            cdf_g_t = self.cdf(g_flattened)
#            cdf_g_t = plt.hist(g_flattened, bins=256, normed=True, cumulative=True)
#            plt.show()
            
            cdf_b_t = self.cdf(b_flattened)
#            cdf_b_t = plt.hist(b_flattened, bins=256, normed=True, cumulative=True)
#            plt.show()
            #-----------------------------------------------------------------------
            
            #--------------inv cat cdf's-------------------------------------------
            foreground = np.logical_or(cat_inv[:,:,1] < 180, cat_inv[:,:,0] > 150)            
            nonzero_x, nonzero_y = np.nonzero(foreground)
            
            r_all = cat_inv[:,:,0]
            g_all = cat_inv[:,:,1]
            b_all = cat_inv[:,:,2]
            
            h,w = cat_inv.shape[:2]
            r_all_flattened = r_all.reshape([h*w])
            g_all_flattened = g_all.reshape([h*w])
            b_all_flattened = b_all.reshape([h*w])
            
            cdf_r = self.cdf(r_all_flattened)
#            cdf_r = plt.hist(r_all_flattened, bins=256, normed=True, cumulative=True)
#            plt.show()
            
            cdf_g = self.cdf(g_all_flattened)
#            cdf_g = plt.hist(g_all_flattened, bins=256, normed=True, cumulative=True)
#            plt.show()
            
            cdf_b = self.cdf(b_all_flattened)
#            cdf_b = plt.hist(b_all_flattened, bins=256, normed=True, cumulative=True)
#            plt.show()
            #----------------------------------------------------------------------
            
            #creating LOOK UP TABLES
            LUT_r = self.LUT(cdf_r, cdf_r_t)
            LUT_g = self.LUT(cdf_g, cdf_g_t)
            LUT_b = self.LUT(cdf_b, cdf_b_t)
            
            #histogram matchings by LUT's
            r_transformed = cv.LUT(r_all, LUT_r) 
            g_transformed = cv.LUT(g_all, LUT_g)
            b_transformed = cv.LUT(b_all, LUT_b)
            
            #merging matched image to the recent background(cat&malibu)
            cat_inv_transformed = cv.merge([r_transformed, g_transformed, b_transformed])
            last_frame = new_frame_single.copy()
            last_frame[nonzero_x, nonzero_y] = cat_inv_transformed[nonzero_x, nonzero_y, :]
            
            last_frame = last_frame[:,:,[2,1,0]] #convert from RGB to BGR
            img_list.append(last_frame) # add the last frame to the image list which will be used to create video
            
            print(f"cat{count} completed!")
            count=count+1
            
        #creating clip and the add audio
        clip = mpy.ImageSequenceClip(img_list, fps=25)
        audio = mpy.AudioFileClip('hw1_material/selfcontrol_part.wav').set_duration(clip.duration)
        clip = clip.set_audio(audioclip=audio)
        clip.write_videofile('part2_video.mp4', codec='libx264')
        print("*****************PART2 COMPLETED!!!*****************")

        
Q1Q2 = HW1_q1q2()
Q1Q2.q1()
Q1Q2.q2()