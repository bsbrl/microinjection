# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:50:51 2021

@author: User
"""

from XYZ_Stage.XYZ_Position import XYZ_Location
from move_embryo_fov_new_new import move_embryo_fov_new_new
import time
from stream_image import stream_image
import cv2
from ML.ml_injection_point_estimation_new import ml_injection_point_estimation_new
import numpy as np
# from open_needle import open_needle

def injection_ml_tip_short_centroid_new(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,X_pos,Y_pos,Z_pos,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic):
    
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
    while end==0 or end==2:
        end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,arduino,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num=move_embryo_fov_new_new(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic)
        if end==2:
            time_wait=.15
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,arduino,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num=move_embryo_fov_new_new(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic)       
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,arduino,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num=move_embryo_fov_new_new(1,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic)
            current_x_centroid=current_x
            current_y_centroid=current_y
            current_z_centroid=current_z
            end=1
        else:
            end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_cen_1,y_coord_emb_cen_1,x_coord_tip_cen_1,y_coord_tip_cen_1,x_coord_emb_cen_2,y_coord_emb_cen_2,x_coord_tip_cen_2,y_coord_tip_cen_2,arduino,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num=move_embryo_fov_new_new(2,'centroid',X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic)       
            current_x_centroid=current_x
            current_y_centroid=current_y
            current_z_centroid=current_z
            end=1
    #subtract dx dy
    dx_move=(current_x_centroid-X_est)
    dy_move=(current_y_centroid-Y_est)
    dx_final=dx_final+dx_move
    dy_final=dy_final+dy_move
    Z_new=next_z
    Z_inj_actual=current_z+300
    img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,0)
    img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop]
    img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop]
    cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Post_injection_new/post_fov_1_{}.jpg'.format(inj_num),img1_crop)
    cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Post_injection_new/post_fov_2_{}.jpg'.format(inj_num),img2_crop)
    # current_z=Z_est-1000
    # XYZ_Location(20000,20000,inj_speed,current_x,current_y,current_z,ser)
    # time.sleep(1.5)
    # # Detect tip
    # tip_x_1=[]
    # tip_y_1=[]
    # tip_x_2=[]
    # tip_y_2=[]
    # img1_list=[]
    # img2_list=[]
    
    # print('detecting')
    # for tip_d in range(10):
    #     img1,img2=stream_image(footage_socket_1,footage_socket_2)
    #     img1_list.append(img1)
    #     img2_list.append(img2)
    # print('ml')
    # output_dict_detection_boxes_stored_pipette_1,output_dict_detection_classes_stored_pipette_1,output_dict_detection_scores_stored_pipette_1,y1a_rc_pipette_1,y2a_rc_pipette_1,x1a_rc_pipette_1,x2a_pipette_1,xc_rc_pipette_1,yc_rc_pipette_1=ml_injection_point_estimation_new(img1_list,.01,720,1280,graph,sess,1)
    # output_dict_detection_boxes_stored_pipette_2,output_dict_detection_classes_stored_pipette_2,output_dict_detection_scores_stored_pipette_2,y1a_rc_pipette_2,y2a_rc_pipette_2,x1a_rc_pipette_2,x2a_rc_pipette_2,xc_rc_pipette_2,yc_rc_pipette_2=ml_injection_point_estimation_new(img2_list,.01,720,1280,graph,sess,1)
    # for k in range(len(output_dict_detection_boxes_stored_pipette_1)):
    #     list_classes_pipette_1=output_dict_detection_classes_stored_pipette_1[k].tolist()
    #     if 5 not in list_classes_pipette_1:
    #         print('Tip not detected FOV1')
    #     else:
    #         list_classes_index_pipette_1=list_classes_pipette_1.index(5)
    #         view_1_x=int(xc_rc_pipette_1[k][list_classes_index_pipette_1])
    #         view_1_y=int(yc_rc_pipette_1[k][list_classes_index_pipette_1])
    #         tip_x_1.append(view_1_x)
    #         tip_y_1.append(view_1_y)
    # for k in range(len(output_dict_detection_boxes_stored_pipette_2)):
    #     list_classes_pipette_2=output_dict_detection_classes_stored_pipette_2[k].tolist()   
    #     if 5 not in list_classes_pipette_2:
    #         print('Tip not detected FOV2')
    #     else:
    #         list_classes_index_pipette_2=list_classes_pipette_2.index(5)
    #         view_2_x=int(xc_rc_pipette_2[k][list_classes_index_pipette_2])
    #         view_2_y=int(yc_rc_pipette_2[k][list_classes_index_pipette_2]) 
    #         tip_x_2.append(view_2_x)
    #         tip_y_2.append(view_2_y)
    # if tip_x_1==[]:
    #     tip_x_1=[view_1_x]
    # if tip_y_1==[]:
    #     tip_y_1=[view_1_y]
    # if tip_x_2==[]:
    #     tip_x_2=[view_2_x]
    # if tip_y_2==[]:
    #     tip_y_2=[view_2_y]
    # view_1_x=int(np.mean(tip_x_1))
    # view_1_y=int(np.mean(tip_y_1))
    # view_2_x=int(np.mean(tip_x_2))
    # view_2_y=int(np.mean(tip_y_2))
    injection_end_time=time.time()
    print('Injection time (s) = ',injection_end_time-injection_start_time)
    return dx_final,dy_final,current_x,current_y,current_z,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,arduino,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid