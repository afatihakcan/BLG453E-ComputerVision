#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 21:25:38 2021

Author: Ahmet Fatih Akcan
Student Number: 150210707

Notes: The script of both part3 and part4 for HW1
"""

import numpy as np
import cv2
import os
import math
import moviepy.editor as moviepy

class HW1_q3q4():
    def __init__(self):
# =============================================================================
#         reads album cover, cat-headphones image, planes and import planes' corner points
# =============================================================================
        self.planes = np.zeros((9,472,4,3))

        self.img = cv2.imread('album_cover.jpg')
        self.cat_headphones = cv2.imread('hw1_material/cat-headphones.png')
        self.cat_headphones = cv2.resize(self.cat_headphones, (572,322))

        self.planes = np.zeros((9,472,4,3))
        for i in range(1,10):
            with open("hw1_material/Plane_"+str(i)+".txt") as f:
                content = f.readlines()
                for line_id in range(len(content)):
                    sel_line = content[line_id]
                    sel_line = sel_line.replace(')\n', '').replace("(", '').split(")")
        
                    for point_id in range(4):
                        sel_point = sel_line[point_id].split(" ")
        
                        self.planes[i-1,line_id,point_id,0] = float(sel_point[0])
                        self.planes[i-1,line_id,point_id,1] = float(sel_point[1])
                        self.planes[i-1,line_id,point_id,2] = float(sel_point[2])
                
    def area(self, frame):
# =============================================================================
#         calculates area of the frames according to corner points
# =============================================================================
        h = abs(frame[1][0][0][0]-frame[1][1][0][0])
        a = abs(frame[1][1][0][1]-frame[1][2][0][1])
        b = abs(frame[1][0][0][1]-frame[1][3][0][1])
        return (a+b)*h/2

    def transformMatrix(self, src, trg):
# =============================================================================
#         calculates transform matrix by least squares estimation
# =============================================================================
        x,y = [src[:,i] for i in range(2)]
        u,v = [trg[:,i] for i in range(2)]
        C = np.zeros(shape=(3,3))
        
        A = np.float64([[x[0], y[0], 1, 0, 0, 0, -x[0]*u[0], -y[0]*u[0]],
                        [x[1], y[1], 1, 0, 0, 0, -x[1]*u[1], -y[1]*u[1]],
                        [x[2], y[2], 1, 0, 0, 0, -x[2]*u[2], -y[2]*u[2]],
                        [x[3], y[3], 1, 0, 0, 0, -x[3]*u[3], -y[3]*u[3]],
                        [0, 0, 0, x[0], y[0], 1, -x[0]*v[0], -y[0]*v[0]],
                        [0, 0, 0, x[1], y[1], 1, -x[1]*v[1], -y[1]*v[1]],
                        [0, 0, 0, x[2], y[2], 1, -x[2]*v[2], -y[2]*v[2]],
                        [0, 0, 0, x[3], y[3], 1, -x[3]*v[3], -y[3]*v[3]]])
        
        B = np.float64([u[0], u[1], u[2], u[3], v[0], v[1], v[2], v[3]]).T
        
        solution = np.linalg.lstsq(A,B)[0]
        
        for i in range(3):
            for j in range(3):
                if i==2 and j==2: 
                    C[i,j] = 1 
                else: 
                    C[i,j] = solution[3*i+j]
                
        return C
    
    def q3(self):
# =============================================================================
#         codes for part3
# =============================================================================
        images_list = []
        frame_list = list()
        
        for i in range(472):
            blank_image = np.zeros((322,572,3), np.uint8)
            blank_image.fill(255)
            frame_list = list()
            for j in range(9):
        
                pts = self.planes[j,i,:,:].squeeze()[:,0:2].astype(np.int32)
        
                temp = np.copy(pts[3,:])
                pts[3, :] = pts[2,:]
                pts[2, :] = temp
        
                pts = pts.reshape((-1, 1, 2))
                
                k1 = np.float64([[0, 0], [self.img.shape[1], 0], [0, self.img.shape[0]], [self.img.shape[1], self.img.shape[0]]])
                k2 = np.float64([pts[0], pts[1], pts[3], pts[2]]).reshape((4,2))
                
                T = self.transformMatrix(k1,k2)
                perspective = cv2.warpPerspective(self.img, T, (572,322))
        
                frame_list.append((perspective, pts))
            
            frame_list.sort(key=self.area) # sort the planes by their area
            
            c=0
            for frame in frame_list:
                if c == 5: #if 5th plane in 9:
                    #add cat-headphones onto the recent bg
                    foreground = np.logical_or(self.cat_headphones[:,:,0] > 0, self.cat_headphones[:,:,1] > 0, self.cat_headphones[:,:,2] > 0)
                    nonzero_x, nonzero_y = np.nonzero(foreground)
                    nonzero_perspective = self.cat_headphones[nonzero_x, nonzero_y, :]
                    blank_image[nonzero_x, nonzero_y] = nonzero_perspective
                    
                frame = frame[0] #rgb values of frame
            
                #mask the cover frame and it onto the recent bg
                foreground = np.logical_or(frame[:,:,0] > 0, frame[:,:,1] > 0, frame[:,:,2] > 0)
                nonzero_x, nonzero_y = np.nonzero(foreground)
                nonzero_perspective = frame[nonzero_x, nonzero_y, :]
                blank_image[nonzero_x, nonzero_y] = nonzero_perspective
                
                c+=1
            
            blank_image = blank_image[:,:,[2,1,0]] # RGB to BGR
            images_list.append(blank_image)
        
        
        clip = moviepy.ImageSequenceClip(images_list, fps = 25)
        clip.write_videofile("part3_video.mp4", codec="libx264")
        print("*****************PART3 COMPLETED!!!*****************")
        
    def q4(self):
# =============================================================================
#         codes for part4
# =============================================================================
        images_list_center = []
        images_list_topLeft = []
        frame_list = list()
        
        for i in range(472):
            blank_image = np.zeros((322,572,3), np.uint8)
            blank_image.fill(255)
            frame_list = list()
            for j in range(9):
        
                pts = self.planes[j,i,:,:].squeeze()[:,0:2].astype(np.int32)
        
                temp = np.copy(pts[3,:])
                pts[3, :] = pts[2,:]
                pts[2, :] = temp
        
                pts = pts.reshape((-1, 1, 2))
                
                k1 = np.float64([[0, 0], [self.img.shape[1], 0], [0, self.img.shape[0]], [self.img.shape[1], self.img.shape[0]]])
                k2 = np.float64([pts[0], pts[1], pts[3], pts[2]]).reshape((4,2))
                
                T = self.transformMatrix(k1,k2)
                perspective = cv2.warpPerspective(self.img, T, (572,322))
        
                frame_list.append((perspective, pts))
            
            frame_list.sort(key=self.area) # sort the planes by their area
            
            
            blank_image_center = blank_image.copy()
            blank_image_topLeft= blank_image.copy()

            c=0
            for frame in frame_list:
                if c == 5: #if 5th plane in 9:
                    #add cat-headphones onto the recent bg
                    foreground = np.logical_or(self.cat_headphones[:,:,0] > 0, self.cat_headphones[:,:,1] > 0, self.cat_headphones[:,:,2] > 0)
                    nonzero_x, nonzero_y = np.nonzero(foreground)
                    nonzero_perspective = self.cat_headphones[nonzero_x, nonzero_y, :]
                    blank_image_center[nonzero_x, nonzero_y] = nonzero_perspective
                    blank_image_topLeft[nonzero_x, nonzero_y] = nonzero_perspective
                    
                frame_center = frame[0]
                frame_topLeft = frame[0]
                
                #rotate 60degrees clockwise from center
                h,w = frame_center.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), -60,1) 
                frame_center = cv2.warpAffine(frame_center, M, (w,h))
                
                #rotate 60degrees clockwise from top left corner
                h,w = frame_center.shape[:2]
                M = cv2.getRotationMatrix2D((0, 0), -60,1)
                frame_topLeft = cv2.warpAffine(frame_topLeft, M, (w,h))
                
                #mask the cover frame(which has been rotated from center) and it onto the recent bg
                foreground = np.logical_or(frame_center[:,:,0] > 0, frame_center[:,:,1] > 0, frame_center[:,:,2] > 0)
                nonzero_x, nonzero_y = np.nonzero(foreground)
                nonzero_perspective = frame_center[nonzero_x, nonzero_y, :]
                blank_image_center[nonzero_x, nonzero_y] = nonzero_perspective
                
                #mask the cover frame(which has been rotated from top left corner) and it onto the recent bg
                foreground = np.logical_or(frame_topLeft[:,:,0] > 0, frame_topLeft[:,:,1] > 0, frame_topLeft[:,:,2] > 0)
                nonzero_x, nonzero_y = np.nonzero(foreground)
                nonzero_perspective = frame_topLeft[nonzero_x, nonzero_y, :]
                blank_image_topLeft[nonzero_x, nonzero_y] = nonzero_perspective
                
                c+=1
            
            blank_image_center = blank_image_center[:,:,[2,1,0]] #RGB to BGR
            blank_image_topLeft = blank_image_topLeft[:,:,[2,1,0]] #RGB to BGR    
            images_list_center.append(blank_image_center)
            images_list_topLeft.append(blank_image_topLeft)
        
        
        clip_center = moviepy.ImageSequenceClip(images_list_center, fps = 25)
        clip_center.write_videofile("part4_video_center.mp4", codec="libx264")
        
        clip_topLeft = moviepy.ImageSequenceClip(images_list_topLeft, fps = 25)
        clip_topLeft.write_videofile("part4_video_topLeft.mp4", codec="libx264")
        print("*****************PART4 COMPLETED!!!*****************")
        
Q3Q4 = HW1_q3q4()
Q3Q4.q3()
Q3Q4.q4()