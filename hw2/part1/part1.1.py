# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Ahmet Fatih Akcan
150210707
"""

import moviepy.video.io.VideoFileClip as mpy
import moviepy.editor as wr
import cv2


vid = mpy.VideoFileClip('shapes_video.mp4')
frame_count = vid.reader.nframes
video_fps = vid.fps

frame_list = []

for i in range(frame_count):
    frame = vid.get_frame(i*1.0/video_fps)
    median = cv2.medianBlur(frame,5) # applying the median filter
    frame_list.append(median) 
    
    
clip = wr.ImageSequenceClip(frame_list, fps=25)
clip.write_videofile('part1.1.mp4', codec='libx264')    