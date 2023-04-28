# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:16:35 2020

@author: enet-joshi317-admin
"""
import cv2
import numpy as np
import glob
import os

def divide_image_yolo(width_image,height_image,filename,dish_number,path):
    folder_names=[glob.glob('D:/Microinjection_Project/Python_Code/ML_Yolo/divided_img/*')]
    for folder in folder_names:
        for f_ in folder:
            os.remove(f_)
    i_=0
    row_num=0
    col_num=0
    img_dish = cv2.imread(path + filename,1)
    for row_image in np.linspace(0,height_image-1000,4):
    #for row_image in np.linspace(0,height_image-400,10):
        row_num+=1
        #for col_image in np.linspace(0,width_image-400,15):
        for col_image in np.linspace(0,width_image-1000,6):
            col_num+=1
            #crop_img_dish = img_dish[int(row_image):int(row_image)+400,int(col_image):int(col_image)+400]
            crop_img_dish = img_dish[int(row_image):int(row_image)+1000,int(col_image):int(col_image)+1000]
            cv2.imwrite('D:/Microinjection_Project/Python_Code/ML_Yolo/divided_img/dish_{}_image_{}_{}_{}.jpg'.format(dish_number,i_+1,row_num,col_num),crop_img_dish)
            i_+=1
        col_num=0
# divide_image(6000,4000,'zebrafish_dish_8.jpg',8,'D:/Microinjection_Project/ML_data/New_data_set/dish_8/')