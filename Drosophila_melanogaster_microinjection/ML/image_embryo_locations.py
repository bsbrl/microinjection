# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:17:42 2022

@author: User
"""

import cv2
import time
import numpy as np
from XYZ_Stage.XYZ_Position import XYZ_Location
import serial

# Connect XYZ stage
ser = serial.Serial('COM3', 9600,timeout = 5)
if not ser.isOpen():
    ser.open()
    
cap_1=cv2.VideoCapture(0)
cap_2=cv2.VideoCapture(1)

cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap_1.set(cv2.CAP_PROP_FPS,30)
cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap_2.set(cv2.CAP_PROP_FPS,30)
cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)

stage_orig=[59370,19640,26195]
delta=[[0,0,0],[25,-10,30],[40,20,-25],[-30,40,-10],[15,-20,10],[-30,-20,-10]]
XYZ_Location(10000,10000,8000,stage_orig[0],stage_orig[1],stage_orig[2],ser)
time.sleep(10)
for i in range(6):
    XYZ_Location(10000,10000,8000,stage_orig[0]+delta[i][0],stage_orig[1]+delta[i][1],stage_orig[2]+delta[i][2],ser)
    time.sleep(1)
    while(True):
        ret_1,frame_1 = cap_1.read()
        ret_2,frame_2 = cap_2.read()
    
        image_height=1280
        image_width=960
        center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
        scale=1
        fromCenter=False
        M_1 = cv2.getRotationMatrix2D(center,450, scale)
        cosine = np.abs(M_1[0, 0])
        sine = np.abs(M_1[0, 1])
        nW = int((image_height * sine) + (image_height * cosine))
        nH = int((image_height * cosine) + (image_width * sine))
        M_1[0, 2] += (nW / 2) - int((float(image_width))/(2))
        M_1[1, 2] += (nH / 2) - int((float(image_height))/(2))
        new_1=cv2.warpAffine(frame_1, M_1, (image_height, image_width)) 
    
        image_height=960
        image_width=1280
        center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
        M_2 = cv2.getRotationMatrix2D(center,180, scale)
        cosine = np.abs(M_2[0, 0])
        sine = np.abs(M_2[0, 1])
        nW = int((image_height * sine) + (image_height * cosine))
        nH = int((image_height * cosine) + (image_width * sine))
        M_2[0, 2] += (nW / 2) - int((float(image_width))/(2))
        M_2[1, 2] += (nH / 2) - int((float(image_height))/(2))
        new_2=cv2.warpAffine(frame_2, M_2, (image_height, image_width)) 
    
        cv2.imshow('view_1',frame_1) #display the captured image
        cv2.imshow('view_2',frame_2) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
         cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/embryo_image_locations/fov_1_{}.jpg'.format(i+1),frame_1)
         cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/embryo_image_locations/fov_2_{}.jpg'.format(i+1),frame_2)
         cv2.destroyAllWindows()
         break
XYZ_Location(10000,10000,8000,stage_orig[0]+delta[i][0],stage_orig[1]+delta[i][1],0,ser)
time.sleep(1)
cap_1.release()
cap_2.release()

