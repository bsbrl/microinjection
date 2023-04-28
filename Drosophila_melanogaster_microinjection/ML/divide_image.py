# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:16:35 2020

@author: enet-joshi317-admin
"""
import cv2
import numpy as np
# def divide_image(width_image,height_image,path_and_filename):
#     i_=0
#     row_num=0
#     col_num=0
#     img_dish = cv2.imread(path_and_filename,1)
#     for row_image in np.linspace(0,height_image-1344,3):
#         row_num+=1
#         for col_image in np.linspace(0,width_image-636,3):
#             col_num+=1
#             crop_img_dish = img_dish[int(row_image):int(row_image)+1344,int(col_image):int(col_image)+636]
#             cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_divided/temp_image_{}_{}_{}.jpg'.format(i_+1,row_num,col_num),crop_img_dish)
#             i_+=1
#         col_num=0
# def divide_image(width_image,height_image,path_and_filename):
#     i_=0
#     row_num=0
#     col_num=0
#     img_dish = cv2.imread(path_and_filename,1)
#     for row_image in np.linspace(0,height_image-1000,4):
#         row_num+=1
#         for col_image in np.linspace(0,width_image-1000,6):
#             col_num+=1
#             crop_img_dish = img_dish[int(row_image):int(row_image)+1000,int(col_image):int(col_image)+1000]
#             cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_divided_bolus/{}_temp_image_{}_{}.jpg'.format(i_+1,row_num,col_num),crop_img_dish)
#             i_+=1
#         col_num=0
def divide_image(width_image,height_image,path_and_filename):
    i_=0
    row_num=0
    col_num=0
    img_dish = cv2.imread(path_and_filename,1)
    for row_image in np.linspace(0,height_image-400,10):
        row_num+=1
        for col_image in np.linspace(0,width_image-400,15):
            col_num+=1
            crop_img_dish = img_dish[int(row_image):int(row_image)+400,int(col_image):int(col_image)+400]
            cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_divided_bolus_paint/{}_temp_image_{}_{}.jpg'.format(i_+1,row_num,col_num),crop_img_dish)
            i_+=1
        col_num=0
