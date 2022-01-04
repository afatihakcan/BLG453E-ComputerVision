#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:36:13 2021

@author: fatih
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os, glob
import imgaug.augmenters as iaa
from torchvision import transforms
import torch
import torchvision.models as models
#import numba
 
def obtain_features(model, img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).cuda()
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)


    return output
    

def prepare_train_from_folder(dir, model): #

    img_count = len(glob.glob(dir +'/*/*.JPEG'))

    train_folders = sorted(os.listdir(dir))

    train = np.zeros((img_count*5, 4096))
    train_labels = np.zeros((img_count*5,))

    counter = 0

    for i in range(len(train_folders)):

        all_imgs = os.listdir(dir + '/' + train_folders[i])

        for j in range(len(all_imgs)):
            img = Image.open(dir + '/' + train_folders[i] + '/' + all_imgs[j])
            img = img.convert('RGB')
            

            try:
                features = obtain_features(model, img)
                train[counter, :] = features.cpu().numpy()
                train_labels[counter] = i
                
            except Exception as e:
                error_list.append(e)
                
            counter += 1
            
            
            im_np = np.asarray(img)
            images = []
            images.append(im_np)
            im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)            
            
            print(f"train preparing: {len(all_imgs)*i+j} in {len(train_folders)*len(all_imgs)}, aug=0")

            for w in range(4):
                augmented_images = aug(images = images)

                for img in augmented_images:
                    im_pil = Image.fromarray(img)
                    im_pil = im_pil.convert('RGB')
                    
                    try:
                        features = obtain_features(model, im_pil)
    
                        train[counter, :] = features.cpu().numpy()
                        train_labels[counter] = i
                        
                    except Exception as e:
                        error_list.append(e)

                counter += 1
                print(f"train preparing: {len(all_imgs)*i+j} in {len(train_folders)*len(all_imgs)}, aug={w+1}")

    return train, train_labels

def prepare_test_from_folder(dir, model):

    img_count = len(glob.glob(dir + '/*.JPEG'))

    test_files = sorted(os.listdir(dir))

    test = np.zeros((img_count, 4096))
    test_names = []

    counter = 0

    for i in range(len(test_files)):
        img = Image.open(dir +  '/' + test_files[i])
        img = img.convert('RGB')

        features = obtain_features(model, img)

        test[counter, :] = features.cpu().numpy()
        counter += 1
        
        test_names.append(test_files[i])
        print(f"test preparing: {i} in { len(test_files)}")

    return test, test_names



aug = iaa.RandAugment(n=2, m=9)

train_dir = 'image-classification/imagenet_50/train'
test_dir = 'image-classification/imagenet_50/test/imgs'
print("starting")
model = torch.load('image-classification/feature_extractor.pth').cuda()
print("model loaded")
model.eval()
print("evaluation completed!")


test, test_names = prepare_test_from_folder(test_dir, model)#

error_list = []
train, train_labels = prepare_train_from_folder(train_dir, model)


from datetime import datetime 

start_time = datetime.now() 
print(f"Start Time (hh:mm:ss.ms): {start_time}")

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Train = sc.fit_transform(train)
del train
print("Train ok!")

Test = sc.transform(test)
del test
print("Test ok!")
        
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=43, n_jobs=-1)
knn.fit(Train, train_labels)
print("KNN fitted!")

del Train
del train_labels

print("Prediction started!")
pred = knn.predict(Test)
print("Prediction completed!")


time_elapsed = datetime.now() - start_time 
print(f"Time elapsed (hh:mm:ss.ms): {time_elapsed}")


pred_values = pred.astype(int)
pred_classes = [train_folders[i] for i in pred_values]

df1 = pd.DataFrame(data=test_names, columns=['FileName'])
df2 = pd.DataFrame(data=pred_classes, columns=['Class'])

df3 = pd.concat([df1, df2], axis=1)

df3.to_csv('evals.csv', index=False)
