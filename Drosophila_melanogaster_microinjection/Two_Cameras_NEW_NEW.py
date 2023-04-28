# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:52:07 2019

@author: Andrew
"""
import cv2
import time
import numpy as np


cap_1=cv2.VideoCapture(1)
cap_2=cv2.VideoCapture(0)

cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap_1.set(cv2.CAP_PROP_FPS,30)
cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap_2.set(cv2.CAP_PROP_FPS,30)
cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)

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

    cv2.circle(frame_1, (560,161), 5, (0, 0, 255) , 2)
    cv2.circle(frame_2, (606,339), 5, (0, 0, 255) , 2)
    
    cv2.imshow('view_1',frame_1) #display the captured image
    cv2.imshow('view_2',frame_2) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
     cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/injection_images/view_1.jpg',frame_1)
     cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/injection_images/view_2.jpg',frame_2)
     cv2.destroyAllWindows()
     break
cap_1.release()
cap_2.release()
