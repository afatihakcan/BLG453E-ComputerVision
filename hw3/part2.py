# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:36:42 2021

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
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # preprocessing for thresholding
            (thresh, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #converting to binary image
            images.append(bw)
            
    return images

def load_maps_from_folder(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        mat = loadmat(os.path.join(folder,filename))
        if mat is not None:
            res = np.zeros(shape=mat['groundTruth'][0][0][0][0][1].shape, dtype = 'uint8') #get boundaries
            for i in range(mat['groundTruth'].shape[1]):
                res = cv2.add(res, mat['groundTruth'][0][i][0][0][1]) #merging groundTruth images
                res[res==1] = 255 #since white means 255 :d
            images.append(res)
            names.append(filename.replace('.mat', '.jpg')) #changing the file extentions
    return images, names


# images, edges = load_images_from_folder('images/test')
edges = load_images_from_folder('part2/output/bsds/test/sing_scale_test')
edge_maps, img_names = load_maps_from_folder('groundTruth/test')

# cv2.imshow("findings", edges[0])
# cv2.imshow("true", edge_maps[0])

# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
    
path = os.getcwd() + '\part2\edges' #path to save edge images
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
    
    cv2.imwrite(f'{img_names[i]}', edges[i])
    
    
print("accuracy= ", round(total/200,2)) # accuracy founded: 44.81%

# edges = cv2.Canny(gray, low_threshold, high_threshold) #detecting edges by Canny Edge detection as a preprocess of line detection 

