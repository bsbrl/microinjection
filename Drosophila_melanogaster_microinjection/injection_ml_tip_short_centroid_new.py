# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:50:51 2021

@author: User
"""

from XYZ_Stage.XYZ_Position import XYZ_Location
from move_embryo_fov_new_new_thresh_pressure import move_embryo_fov_new_new_thresh_pressure
import time
from stream_image import stream_image
import cv2
from ML.ml_injection_point_estimation_new import ml_injection_point_estimation_new
import numpy as np
# from open_needle import open_needle

def injection_ml_tip_short_centroid_new(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,X_pos,Y_pos,Z_pos,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic):
    
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
    next_z=Z_est
    move_num=2
    thresh_1=.01
    thresh_2=.01
    
    print('Current X = ',X_est)
    print('Current Y = ',Y_est)
    print('Current Z = ',Z_est)
    end=0
    
    x1_1_crop=int(view_1_x-400)
    x2_1_crop=int(view_1_x+400)
    y1_1_crop=int(view_1_y-300)
    y2_1_crop=int(view_1_y+300)
       

    x1_2_crop=int(view_2_x-400)
    x2_2_crop=int(view_2_x+400)
    y1_2_crop=int(view_2_y-300)
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
    time.sleep(time_wait)
    while end!=3:
        end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'no move centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,0,0,0,0,0,0,1)        
        current_x_centroid=current_x
        current_y_centroid=current_y
        current_z_centroid=current_z
        if end_1!=1 and end_2!=1:
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(0,'just move',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,0,0,0,0,0,0,0)
            end=3
        else:
            end==3

    # while end!=3:
    #     end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,0,0,0,0,0,0,0)
    #     if end==2:
    #         time_wait=.15
    #         end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,0,0,0,0,0,0,0)       
    #         end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,0,0,0,0,0,0,0)
    #         if end==2:
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,1,0,0,0,0,0,0,0)           
    #         current_x_centroid=current_x
    #         current_y_centroid=current_y
    #         current_z_centroid=current_z
    #         end=3
    #     else:
    #         end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,0,0,0,0,0,0,0,0) 
    #         if end==1:
    #             end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num,injected,end_1,end_2,injected,sum_image,pressure_value=move_embryo_fov_new_new_thresh_pressure(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic,thresh_1,thresh_2,2,0,0,0,0,0,0,0)           
    #         current_x_centroid=current_x
    #         current_y_centroid=current_y
    #         current_z_centroid=current_z
    #         end=3
    #subtract dx dy
    dx_move=(current_x_centroid-X_est)
    dy_move=(current_y_centroid-Y_est)
    dx_final=dx_final+dx_move
    dy_final=dy_final+dy_move
    Z_new=next_z
    Z_inj_actual=current_z+300

    injection_end_time=time.time()
    print('Injection time (s) = ',injection_end_time-injection_start_time)
    return dx_final,dy_final,current_x,current_y,current_z,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,arduino,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid