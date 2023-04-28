# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:37:58 2021

@author: Andrew
"""

import cv2


def click_event(event, x, y, flags, params): 
    global ix,iy
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ix,iy=x,y  
    # checking for right mouse clicks      
    if event==cv2.EVENT_RBUTTONDOWN: 
        ix,iy=x,y 
def point_and_click(img):
    # img=cv2.imread(image_path)
    print('Click then press y')
    cv2.imshow('image', img) 
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    x=int(ix)
    y=int(iy)
    cv2.circle(img, (x,y), radius=5, color=(0, 0, 255), thickness=1)
    print('Check then press y')
    cv2.imshow('image', img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    return x,y
# x,y=point_and_click('C:/Users/Andrew/anaconda3/Annotations_Robot_2/Annotations_Robot_2/only_images/pipette_detection_fov_1_1_1.jpg')