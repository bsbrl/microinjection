# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:58:00 2022

@author: User
"""

import cv2
import numpy as np

def detect_injection(img,x_post,y_post,inj_num,fov):
    lower_blue = np.array([52,73,35])
    upper_blue = np.array([255,255,255])
    
    # img_crop=img[int(y_post-30):int(y_post+40),int(x_post-40):int(x_post+40)]
    img_crop=img[int(y_post-20):int(y_post+30),int(x_post-30):int(x_post+30)]
    # img_crop=img
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    sum_image=sum(sum(mask))
    # print(sum_image)
    # cv2.imshow('mask',mask)
    # cv2.waitKey(0)
    
    # if fov==1:
    #     cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/detection_injection_images_new/{}_embryo_fov_1_image_1.jpg'.format(inj_num+1),img)
    #     cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/detection_injection_images_new/{}_embryo_fov_1_image_2.jpg'.format(inj_num+1),img_crop)
    # else:
    #     cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/detection_injection_images_new/{}_embryo_fov_2_image_1.jpg'.format(inj_num+1),img)
    #     cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/detection_injection_images_new/{}_embryo_fov_2_image_2.jpg'.format(inj_num+1),img_crop)
    
    return sum_image
# x_post=595
# y_post=290
# img1 = cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/detection_injection_images_new_new/45_embryo_fov_1_image_1.jpg',1)
# sum_image=detect_injection(img1,x_post,y_post,2,2)
