# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:05:29 2021

@author: User
"""

from XYZ_Stage.XYZ_Position import XYZ_Location
# from Two_Cameras_Video import Two_Cameras_Video
# import cv2 
from ML.ml_injection_point_estimation_new import ml_injection_point_estimation_new
import time
from stream_image import stream_image

# import numpy as np

def new_z(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,X_pos,Y_pos,Z_pos,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,pip_num,Z_inj,pic):
    
    new_pipette_start_time=time.time()
    X_est=X_pos
    Y_est=Y_pos
    Z_est=Z_pos
    Z_est_old=Z_pos
    
    print('Current X = ',X_est)
    print('Current Y = ',Y_est)
    print('Current Z = ',Z_est)
    print('New Pipette')
    print('TURN VALVES!')
    XYZ_Location(20000,20000,8000,X_est,Y_est,5000,ser)
    time.sleep(10)

    x1_1_crop=int(view_1_x-400)
    x2_1_crop=int(view_1_x+400)
    # y1_1_crop=int(view_1_y-300)
    y1_1_crop=int(view_1_y-50)
    y2_1_crop=int(view_1_y+300)
       

    x1_2_crop=int(view_2_x-400)
    x2_2_crop=int(view_2_x+400)
    # y1_2_crop=int(view_2_y-300)
    y1_2_crop=int(view_2_y-50)
    y2_2_crop=int(view_2_y+300)

    if x1_1_crop<0:
        x1_1_crop=0
    if y1_1_crop<0:
        y1_1_crop=0
    if x2_1_crop>1280:
        x2_1_crop=1280    
    if y2_1_crop>720:
        y2_1_crop=720
        
    if x1_2_crop<0:
        x1_2_crop=0
    if y1_2_crop<0:
        y1_2_crop=0
    if x2_2_crop>1280:
        x2_2_crop=1280
    if y2_2_crop>720:
        y2_2_crop=720
        
    im_width_1=x2_1_crop-x1_1_crop
    im_height_1=y2_1_crop-y1_1_crop 
    im_width_2=x2_2_crop-x1_2_crop
    im_height_2=y2_2_crop-y1_2_crop

    s_end=0
    s=0
    XYZ_Location(5000,5000,2000,X_est,Y_est,Z_est,ser)
    time.sleep(10)
    while s_end==0 and Z_est<Z_inj+500:
        Z_est=Z_est_old+100*s
        XYZ_Location(5000,5000,2000,X_est,Y_est,Z_est,ser)
        time.sleep(.5)
        img1,img2=stream_image(footage_socket_1,footage_socket_2,footage_socket_3,pic,0)
        img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop]
        img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop]
        output_dict_detection_boxes_stored_1,output_dict_detection_classes_stored_1,output_dict_detection_scores_stored_1,y1a_rc_1,y2a_rc_1,x1a_rc_1,x2a_rc_1,xc_rc_1,yc_rc_1=ml_injection_point_estimation_new([img1_crop],.1,im_height_1,im_width_1,graph,sess,1)
        list_classes_1_c=output_dict_detection_classes_stored_1[0].tolist()
        output_dict_detection_boxes_stored_2,output_dict_detection_classes_stored_2,output_dict_detection_scores_stored_2,y1a_rc_2,y2a_rc_2,x1a_rc_2,x2a_rc_2,xc_rc_2,yc_rc_2=ml_injection_point_estimation_new([img2_crop],.1,im_height_2,im_width_2,graph,sess,1)
        list_classes_2_c=output_dict_detection_classes_stored_2[0].tolist()
        if 1 not in list_classes_1_c and 1 not in list_classes_2_c:
            print('No centroid detected FOV1')   
            Z_new=Z_est+100
            s+=1  
        else:
            print('Embryo in FOV')
            print('Initial Z estimate = ',Z_est+100)
            Z_new=Z_est+100
            XYZ_Location(5000,5000,2000,X_est,Y_est,Z_new,ser)
            time.sleep(.5)
            s_end=1
    new_pipette_end_time=time.time()
    print('New pipette time (s) = ',new_pipette_end_time-new_pipette_start_time)
    return Z_new