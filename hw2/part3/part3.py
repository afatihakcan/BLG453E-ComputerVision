# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 20:43:11 2021

@author: Ahmet Fatih Akcan
"""

import pyautogui
import time
import cv2
import numpy as np

length = 250 # x-axis length to calculate lines of corresponding shape
rect_pts = [[812, 869], [812, 1051], [1078, 869], [1078, 1051]] #y,x

low_threshold = 50 # for Canny edge detection
high_threshold = 150 # for Canny edge detection

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

time.sleep(5)
is_first = True # new shape flag

while(1):
    ss = pyautogui.screenshot()
    ss = np.asarray(ss)
    (x,y,_) = np.nonzero(ss[:,:] == [204,204,224]) #masking the pressing zone..
    ss[x,y, :] = 255  #and making this zone white to prevent the conflict while line detection
    
    #Getting rid of the RGB dimensions.....
    gray = cv2.cvtColor(ss, cv2.COLOR_BGR2GRAY)
    gray = gray[rect_pts[0][0]:rect_pts[2][0], rect_pts[0][1]:rect_pts[0][1]+length]
    gray = cv2.erode(gray,np.ones((5,5),np.uint8),iterations = 1) #to make edges better for line detection (actually I just wanted to try it as an idea after the code is done and working, and it just worked fine!)
    gray = cv2.GaussianBlur(gray,(5, 5),0) #also this.... (but I suppose not as effective as the erosion...)
    (thresh, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #binary thresholding candır 
    # bw = cv2.GaussianBlur(bw,(15, 15),0) # çok da abartmayalım :d
    
    edges = cv2.Canny(gray, low_threshold, high_threshold) #detecting edges by Canny Edge detection as a preprocess of line detection 
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap) # magical line detection

    check_arr = bw[:, 10] # an array to check if a new shape is arrived to the 10px to the border of the pressing zone
    if np.count_nonzero(check_arr == 0) and is_first: #if bir cisim yaklaşıyor efendim and it is a new shape
        is_first = False # not new anymore
        
        if lines.shape[0] > 11: # experimental values which indicates to the star shape
            # pyautogui.press('d')
            pyautogui.keyDown('d') #press the key
            time.sleep(0.3) #hold it for 0.3 seconds
            pyautogui.keyUp('d') #release the key
            
        elif lines.shape[0] == 2 or lines.shape[0] == 3 or lines.shape[0] == 4: # experimental values which indicates to the square shape
            # pyautogui.press('s')
            pyautogui.keyDown('s') #press the key
            time.sleep(0.3) #hold it for 0.3 seconds
            pyautogui.keyUp('s') #release the key
            
            
        elif lines.shape[0] <= 8 and lines.shape[0] > 5: # experimental values which indicates to the triangle shape
            # pyautogui.press('a')
            pyautogui.keyDown('a') #press the key
            time.sleep(0.3) #hold it for 0.3 seconds
            pyautogui.keyUp('a') #release the key
            
            
        elif lines.shape[0] == 9 or lines.shape[0] == 10:  # experimental value which indicates to the hexagon shape
            # pyautogui.press('f')
            pyautogui.keyDown('f') #press the key
            time.sleep(0.3) #hold it for 0.3 seconds
            pyautogui.keyUp('f') #release the key
            
            
        else:
            print("error: ", lines.shape[0]) # thankfully only happened while not playing the game :))
        
    elif not np.count_nonzero(check_arr == 0): # there is not a shape into the region anymore
        is_first = True # gelirse bizimdir gelmezse oyunu bitirmişizdir 
        
    
    cv2.imshow("ss", ss[ rect_pts[0][0]:rect_pts[2][0] ,  rect_pts[0][1]:rect_pts[1][1] , :])
    cv2.imshow("gray", gray)
    cv2.imshow("bw", bw)
    if cv2.waitKey(1) & 0xff == 27:
        cv2.destroyAllWindows()
        break