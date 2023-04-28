# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:05:10 2022

@author: User
"""


from XYZ_Stage.XYZ_Position import XYZ_Location
from move_embryo_fov_new_new_thresh_pressure import move_embryo_fov_new_new_thresh_pressure
import time
from stream_image import stream_image
import cv2
from ML.ml_injection_point_estimation_new import ml_injection_point_estimation_new
import numpy as np

def injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,X_pos,Y_pos,Z_pos,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel,miss):
    injection_start_time=time.time()
    injection_list_num=0
    X_est=X_pos
    Y_est=Y_pos
    Z_est=Z_pos
    current_x=X_est
    current_y=Y_est
    current_z=Z_est
    current_x_centroid=X_est
    current_y_centroid=Y_est
    current_z_centroid=Z_est
    current_z_new=0
    x_coord_emb_1=0
    y_coord_emb_1=0
    x_coord_emb_2=0
    y_coord_emb_2=0
    next_z=Z_est
    move_num=2
    thresh_1=.01
    thresh_2=.01
    print('Current X = ',X_est)
    print('Current Y = ',Y_est)
    print('Current Z = ',Z_est)
    end=0
    injection_time=0
    
    x1_1_crop=int(view_1_x-250)
    x2_1_crop=int(view_1_x+250)
    y1_1_crop=int(view_1_y-25)
    y2_1_crop=int(view_1_y+300)
       

    x1_2_crop=int(view_2_x-250)
    x2_2_crop=int(view_2_x+250)
    y1_2_crop=int(view_2_y-25)
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
    XYZ_Location(20000,20000,8000,X_est,Y_est,Z_est,ser)
    time.sleep(1)

    while end!=3:
        move_num=2
        diff=0
        if miss==1:
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,1)        
            current_x_centroid=current_x
            current_y_centroid=current_y
            current_z_centroid=current_z
            if end_1!=1 and end_2!=1:
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
            else:
                end==3
        while diff==0 and move_num<7:
            if move_num==2:
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,1)       
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
            if end_1==1:
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
                if end_1==1:
                    end=3
            if end_2==1:
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)                    
                if end_2==1:
                    end=3
            print('FOV 1 diff = ',abs(x_coord_emb_1-x_coord_tip_1))
            print('FOV 2 diff = ',abs(x_coord_emb_2-x_coord_tip_2))
            # if abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num:
            if abs(x_coord_emb_1-x_coord_tip_1)<=10 and abs(x_coord_emb_2-x_coord_tip_2)<=10:
                diff=1
            else:
                diff=0
            move_num+=1
            print('Move = ',move_num)
        if diff==0:
            end=3
        else:
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
            if end_1==1:
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
                if end_1==1:
                    end=3
            if end_2==1:
                end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
                if end_2==1:
                    end=3 
    # while end!=3:
    #     if miss==1:
    #         end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #     if end==2 and miss==1:
    #         end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #         end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #         if end==2 and miss==1:
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)           
    #             if end==2 and miss==1:
    #                 current_x_centroid=current_x
    #                 current_y_centroid=current_y
    #                 current_z_centroid=current_z
    #                 end=3
    #         else:
    #             move_num=2
    #             diff=0
    #             current_x_centroid=current_x
    #             current_y_centroid=current_y
    #             current_z_centroid=current_z
    #             while diff==0 and move_num<7:
    #                 if move_num==2:
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,1)       
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #                 if end_1==1:
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                     if end_1==1:
    #                         end=3
    #                 if end_2==1:
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                     if end_2==1:
    #                         end=3
    #                 print('FOV 1 diff = ',abs(x_coord_emb_1-x_coord_tip_1))
    #                 print('FOV 2 diff = ',abs(x_coord_emb_2-x_coord_tip_2))
    #                 # if abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num:
    #                 if abs(x_coord_emb_1-x_coord_tip_1)<=10 and abs(x_coord_emb_2-x_coord_tip_2)<=10:
    #                     diff=1
    #                 else:
    #                     diff=0
    #                     move_num+=1
    #                 print('Move = ',move_num)
    #             if diff==0:
    #                 end=3
    #             else:
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                 if end_1==1:
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                     if end_1==1:
    #                         end=3
    #                 if end_2==1:
    #                     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                     if end_2==1:
    #                         end=3                       
    #     else:
    #         move_num=2
    #         diff=0
    #         if miss==1:
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #         if end==1 and miss==1:
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)           
    #             if end==1 and miss==1:
    #                 current_x_centroid=current_x
    #                 current_y_centroid=current_y
    #                 current_z_centroid=current_z
    #                 end=3
    #         current_x_centroid=current_x
    #         current_y_centroid=current_y
    #         current_z_centroid=current_z
    #         while diff==0 and move_num<7:
    #             if move_num==2:
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,1)       
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #             if end_1==1:
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                 if end_1==1:
    #                     end=3
    #             if end_2==1:
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)                    
    #                 if end_2==1:
    #                     end=3
    #             print('FOV 1 diff = ',abs(x_coord_emb_1-x_coord_tip_1))
    #             print('FOV 2 diff = ',abs(x_coord_emb_2-x_coord_tip_2))
    #             # if abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num:
    #             if abs(x_coord_emb_1-x_coord_tip_1)<=10 and abs(x_coord_emb_2-x_coord_tip_2)<=10:
    #                 diff=1
    #             else:
    #                 diff=0
    #             move_num+=1
    #             print('Move = ',move_num)
    #         if diff==0:
    #             end=3
    #         else:
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)       
    #             if end_1==1:
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                 if end_1==1:
    #                     end=3
    #             if end_2==1:
    #                 end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_inj_1,y_coord_emb_inj_1,x_coord_tip_inj_1,y_coord_tip_inj_1,x_coord_emb_inj_2,y_coord_emb_inj_2,x_coord_tip_inj_2,y_coord_tip_inj_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'inject new',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,sum_image_thresh_min,target_pixel,x_coord_emb_1,y_coord_emb_1,x_coord_emb_2,y_coord_emb_2,0)
    #                 if end_2==1:
    #                     end=3 
    injection_end_time=time.time()
    print('Injection time (s) = ',injection_end_time-injection_start_time-2.5)
    if pic%5!=0 and sum_image_thresh_min<sum_image:
        injection_time=injection_end_time-injection_start_time-2.5
    #subtract dx dy
    dx_move=(current_x_centroid-X_est)
    dy_move=(current_y_centroid-Y_est)
    dx_final=dx_final+dx_move
    dy_final=dy_final+dy_move
    Z_inj_actual=current_z+300
    if injection_list_num==1 or injection_list_num==2:
        # Z_new=next_z
        Z_new=Z_inj_actual-330
    else:
        Z_new=Z_est 
    # if sum_image_thresh_min>sum_image:
    if abs(x_coord_emb_1-x_coord_tip_1)>10 or abs(x_coord_emb_2-x_coord_tip_2)>10:
        miss=1
        current_z=Z_est-1000
        XYZ_Location(20000,20000,8000,current_x,current_y,current_z,ser)
        time.sleep(.5)
        # Detect tip
        tip_x_1=[]
        tip_y_1=[]
        tip_x_2=[]
        tip_y_2=[]
        
        lower_blue = np.array([52,30,35])
        upper_blue = np.array([255,255,255])
        img1,img2=stream_image(footage_socket_1,footage_socket_2,footage_socket_3,pic,0)
        # cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/Test_pipette/FOV_1_{}.jpg'.format(inj_num),img1)
        # cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/Test_pipette/FOV_2_{}.jpg'.format(inj_num),img2)
        output_dict_detection_boxes_stored_pipette_1,output_dict_detection_classes_stored_pipette_1,output_dict_detection_scores_stored_pipette_1,y1a_rc_pipette_1,y2a_rc_pipette_1,x1a_rc_pipette_1,x2a_rc_pipette_1,xc_rc_pipette_1,yc_rc_pipette_1=ml_injection_point_estimation_new([img1],.01,720,1280,graph,sess,1)
        output_dict_detection_boxes_stored_pipette_2,output_dict_detection_classes_stored_pipette_2,output_dict_detection_scores_stored_pipette_2,y1a_rc_pipette_2,y2a_rc_pipette_2,x1a_rc_pipette_2,x2a_rc_pipette_2,xc_rc_pipette_2,yc_rc_pipette_2=ml_injection_point_estimation_new([img2],.01,720,1280,graph,sess,1)
        list_classes_pipette_1=output_dict_detection_classes_stored_pipette_1[0].tolist()
        if 5 in list_classes_pipette_1:
            list_classes_index_pipette_1=list_classes_pipette_1.index(5)
            crop_1=img1[y1a_rc_pipette_1[0][list_classes_index_pipette_1]:y2a_rc_pipette_1[0][list_classes_index_pipette_1],x1a_rc_pipette_1[0][list_classes_index_pipette_1]:x2a_rc_pipette_1[0][list_classes_index_pipette_1]]
            hsv_1 = cv2.cvtColor(crop_1, cv2.COLOR_BGR2HSV)
            mask_1 = cv2.inRange(hsv_1, lower_blue, upper_blue)
            mask_1_list=mask_1.tolist()
            x_list=[]
            y_list=[]
            for j in range(len(mask_1)):
                indices = [i for i, x in enumerate(mask_1_list[j]) if x==255]
                if indices!=[]:
                    x_list.append(np.median(indices))
                    y_list.append(j)
            if x_list==[] or y_list==[] or x1a_rc_pipette_1==[] or y1a_rc_pipette_1==[]:
                tip_x_1.append(view_1_x)
                tip_y_1.append(view_1_y)
                print('CV tip x = ',view_1_x)
                print('CV tip y = ',view_1_y)
            else:
                tip_x_1.append(int(x_list[len(x_list)-1]+x1a_rc_pipette_1[0][list_classes_index_pipette_1]))
                tip_y_1.append(int(y_list[len(y_list)-1]+y1a_rc_pipette_1[0][list_classes_index_pipette_1]))
                print('CV tip x = ',int(x_list[len(x_list)-1]+x1a_rc_pipette_1[0][list_classes_index_pipette_1]))
                print('CV tip y = ',int(y_list[len(y_list)-1]+y1a_rc_pipette_1[0][list_classes_index_pipette_1]))
        list_classes_pipette_2=output_dict_detection_classes_stored_pipette_2[0].tolist()
        if 5 in list_classes_pipette_2:
            list_classes_index_pipette_2=list_classes_pipette_2.index(5)
            crop_2=img2[y1a_rc_pipette_2[0][list_classes_index_pipette_2]:y2a_rc_pipette_2[0][list_classes_index_pipette_2],x1a_rc_pipette_2[0][list_classes_index_pipette_2]:x2a_rc_pipette_2[0][list_classes_index_pipette_2]]
            hsv_2 = cv2.cvtColor(crop_2, cv2.COLOR_BGR2HSV)
            mask_2 = cv2.inRange(hsv_2, lower_blue, upper_blue)
            mask_2_list=mask_2.tolist()
            x_list=[]
            y_list=[]
            for j in range(len(mask_2)):
                indices = [i for i, x in enumerate(mask_2_list[j]) if x==255]
                if indices!=[]:
                    x_list.append(np.median(indices))
                    y_list.append(j)
            if x_list==[] or y_list==[] or x1a_rc_pipette_2==[] or y1a_rc_pipette_2==[]:
                tip_x_2.append(view_2_x)
                tip_y_2.append(view_2_y)
                print('CV tip x = ',view_2_x)
                print('CV tip y = ',view_2_y)
            else:
                tip_x_2.append(int(x_list[len(x_list)-1]+x1a_rc_pipette_2[0][list_classes_index_pipette_2]))
                tip_y_2.append(int(y_list[len(y_list)-1]+y1a_rc_pipette_2[0][list_classes_index_pipette_2]))
                print('CV tip x = ',int(x_list[len(x_list)-1]+x1a_rc_pipette_2[0][list_classes_index_pipette_2]))
                print('CV tip y = ',int(y_list[len(y_list)-1]+y1a_rc_pipette_2[0][list_classes_index_pipette_2]))
        if tip_x_1==[]:
            tip_x_1=[view_1_x]
        if tip_y_1==[]:
            tip_y_1=[view_1_y]
        if tip_x_2==[]:
            tip_x_2=[view_2_x]
        if tip_y_2==[]:
            tip_y_2=[view_2_y]
        view_1_x=int(np.mean(tip_x_1))
        view_1_y=int(np.mean(tip_y_1))
        view_2_x=int(np.mean(tip_x_2))
        view_2_y=int(np.mean(tip_y_2))
    else:
        miss=0
        
    return dx_final,dy_final,current_x,current_y,current_z,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value,injection_time,miss

    