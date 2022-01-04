# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 22:51:00 2021

@author: Ahmet Fatih Akcan
Student Number: 150210707
"""

import cv2
import os
import glob
# from mat4py import loadmat
from scipy.io import loadmat
import numpy as np

def load_images_from_folder(folder):
    images = []
    edges = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        names.append(filename)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 50, 150)             
            edges.append(edge)
            images.append(img)
    return images, edges, names

def load_maps_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        mat = loadmat(os.path.join(folder,filename))
        if mat is not None:
            res = np.zeros(shape=mat['groundTruth'][0][0][0][0][1].shape, dtype = 'uint8') #get boundaries
            for i in range(mat['groundTruth'].shape[1]):
                res = cv2.add(res, mat['groundTruth'][0][i][0][0][1]) #merging groundTruth images
                res[res==1] = 255 #since white means 255 :d
            images.append(res)
    return images


images, edges, src_names = load_images_from_folder('images/test')
edge_maps = load_maps_from_folder('groundTruth/test')

# cv2.imshow("findings", edges[0])
# cv2.imshow("true", edge_maps[0])

# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
    

path = os.getcwd() + '\edges' #path to save edge images
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

os.chdir(path)

total = 0
for i in range(200):
    intersection = np.count_nonzero(np.bitwise_and(edges[i], edge_maps[i])) # bitwise and operator to find intersection between findings and actual edges
    # res = np.count_nonzero(edges[i])
    true = np.count_nonzero(edge_maps[i])
    total += intersection/true * 100 #comparing the intersection and the groundTruths for accuracy
    
    cv2.imwrite(f'{src_names[i]}', edges[i])

    
print("accuracy= ", round(total/200,2)) #accuracy founded: 28.3%

# edges = cv2.Canny(gray, low_threshold, high_threshold) #detecting edges by Canny Edge detection as a preprocess of line detection 

