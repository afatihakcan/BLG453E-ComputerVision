#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
from torchvision import transforms
import torch
import torchvision.models as models
import os, glob
import numpy as np
#import pyvips

#"""
#def usingVIPS(f): 
#    image = pyvips.Image.new_from_file(f, access="sequential") 
#    mem_img = image.write_to_memory()
#    try:
#        imgnp=np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width, 3)  
#    except ValueError:
#        imgnp=np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width)
#    return imgnp 
#"""

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
"""
def obtain_features_grayscale(model, img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, ),(0,456)),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)


    return output
    


def prepare_train_from_folder(dir, model):    
    img_count = len(glob.glob(dir +'/*/*.JPEG'))

    train_folders = sorted(os.listdir(dir))

    train = np.zeros((img_count, 4096))
    train_labels = np.zeros((img_count,))

    counter = 0

    for i in range(len(train_folders)):
        all_imgs = os.listdir(dir + '/' + train_folders[i])

        for j in range(len(all_imgs)):
#            img = Image.open(dir + '/' + train_folders[i] + '/' + all_imgs[j])
            
            img = usingVIPS(dir + '/' + train_folders[i] + '/' + all_imgs[j])
#            img = img.convert('RGB')
            try:
                img = Image.fromarray(img, "RGB")
                features = obtain_features(model, img)
            except ValueError:
                print("now!")
                img = Image.fromarray(img, "L")
                features = obtain_features_grayscale(model, img)

#            features = obtain_features(model, img)

            train[counter, :] = features
            train_labels[counter] = i

            counter += 1
            print(f"train preparing: {len(all_imgs)*i+j} in {len(train_folders)*len(all_imgs)}")

    return train, train_labels


def prepare_test_from_folder(dir, model):

    img_count = len(glob.glob(dir + '/*.JPEG'))

    test_files = sorted(os.listdir(dir))

    test = np.zeros((img_count, 4096))
    test_names = []

    counter = 0

    for i in range(len(test_files)):
#        img = Image.open(dir +  '/' + test_files[i])
        img = usingVIPS(dir +  '/' + test_files[i])
#        img = img.convert('RGB')
        try:
            img = Image.fromarray(img, "RGB")
            features = obtain_features(model, img)
        except ValueError:
            img = Image.fromarray(img, "L")
            features = obtain_features_grayscale(model, img)

#        features = obtain_features(model, img)

        test[counter, :] = features

        test_names.append(test_files[i])
        print(f"test preparing: {i} in { len(test_files)}")

    return test, test_names
"""

def prepare_train_from_folder(dir, model):

    img_count = len(glob.glob(dir +'/*/*.JPEG'))

    train_folders = sorted(os.listdir(dir))

    train = np.zeros((img_count, 4096))
    train_labels = np.zeros((img_count,))

    counter = 0

    for i in range(len(train_folders)):
        all_imgs = os.listdir(dir + '/' + train_folders[i])

        for j in range(len(all_imgs)):
            img = Image.open(dir + '/' + train_folders[i] + '/' + all_imgs[j])
            img = img.convert('RGB')
            
            features = obtain_features(model, img)

            train[counter, :] = features.cpu().numpy()
            train_labels[counter] = i

            counter += 1
            print(f"train preparing: {len(all_imgs)*i+j} in {len(train_folders)*len(all_imgs)}")

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

        test_names.append(test_files[i])
        print(f"test preparing: {i} in { len(test_files)}")

    return test, test_names


train_dir = 'image-classification/imagenet_50/train'
test_dir = 'image-classification/imagenet_50/test/imgs'
print("starting")
model = torch.load('image-classification/feature_extractor.pth').cuda()
print("model loaded")
model.eval()
print("evaluation completed!")


test, test_names = prepare_test_from_folder(test_dir, model)

train, train_labels = prepare_train_from_folder(train_dir, model)

#Prepare train_extra and train_extra labels using also prepare_train_from_folder function.
