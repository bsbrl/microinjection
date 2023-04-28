# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:11:58 2021

@author: User
"""
from Delta_XY_FOV_1_lin import Delta_XY_FOV_1_lin
from Delta_XY_FOV_2_lin import Delta_XY_FOV_2_lin
from Delta_Z_FOV_1_lin import Delta_Z_FOV_1_lin
from Delta_Z_FOV_2_lin import Delta_Z_FOV_2_lin
import numpy as np
import time
from stream_image import stream_image
from ML.ml_injection_point_estimation_new_scores import ml_injection_point_estimation_new_scores
import cv2
from Pressure_Control.Continuous_Pressure import continuous_pressure
from XYZ_Stage.XYZ_Position import XYZ_Location
from matplotlib import pyplot as plt
from ML.transformation_matrix_embryo_guess import function_transformation_matrix_embryo_guess

def find_posterior_new(current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,footage_socket_1,footage_socket_2,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,ser,inj_num,pic):
    # time.sleep(.15)
    img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,0)
    img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop] 
    img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop]
    # img1_crop=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/Detect_injection/post_fov_1_531.jpg',1)
    # img2_crop=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/Detect_injection/post_fov_2_531.jpg',1)
    output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc_1,y2a_rc_1,x1a_rc_1,x2a_rc_1,xc_rc_1,yc_rc_1,scores_list_1=ml_injection_point_estimation_new_scores([img1_crop],.0001,im_height_1,im_width_1,graph,sess,3)
    list_classes_1=output_dict_detection_classes_stored[0].tolist() 
    output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc_2,y2a_rc_2,x1a_rc_2,x2a_rc_2,xc_rc_2,yc_rc_2,scores_list_2=ml_injection_point_estimation_new_scores([img2_crop],.0001,im_height_2,im_width_2,graph,sess,2)
    list_classes_2=output_dict_detection_classes_stored[0].tolist()
    cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/images/image_FOV_1_{}.jpg'.format(inj_num),img1)
    cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/images/image_FOV_2_{}.jpg'.format(inj_num),img2)
    end=0
    q='x'
    q_o='x'
    q_e='1'
    press_count=1
    o=0
    injection_list_num=0
    back_pressure_value_new=pressure_value
    current_x_new=current_x
    current_y_new=current_y
    current_z_new=current_z
    
    if 3 in list_classes_1 and 3 in list_classes_2 and 1 in list_classes_1 and 1 in list_classes_2:
        print('m')
        list_classes_index_1_post=list_classes_1.index(3)
        list_classes_index_1_cen=list_classes_1.index(1)
        list_classes_index_2_post=list_classes_2.index(3)
        list_classes_index_2_cen=list_classes_2.index(1)
        x_post_1=xc_rc_1[0][list_classes_index_1_post]
        x_post_2=xc_rc_2[0][list_classes_index_2_post]
        y_post_1=yc_rc_1[0][list_classes_index_1_post]
        y_post_2=yc_rc_2[0][list_classes_index_2_post]    
        x_cen_1=xc_rc_1[0][list_classes_index_1_cen]
        x_cen_2=xc_rc_2[0][list_classes_index_2_cen]
        y_cen_1=yc_rc_1[0][list_classes_index_1_cen]
        y_cen_2=yc_rc_2[0][list_classes_index_2_cen]
        if scores_list_1[0][list_classes_index_1_post]>scores_list_2[0][list_classes_index_2_post]:
            if x_cen_2==x_post_2:
                print('No posterior detected FOV2')
            else: 
                m = (y_cen_2-y_post_2)/(x_cen_2-x_post_2)
                b = (x_cen_2*y_post_2 - x_post_2*y_cen_2)/(x_cen_2-x_post_2)
                x_ant_2=x_cen_2+(x_cen_2-x_post_2)
                y_ant_2 = m*x_ant_2 + b
                dx_2=(x_cen_2-x_post_2)
                dy_2=(y_cen_2-y_post_2)
                dx_1=(x_cen_1-x_post_1)
                dy_1=(y_cen_1-y_post_1)
                if dx_1<0 and dy_1>0 and dx_2>0 and dy_2>0 or dx_1<0 and dy_1<0 and dx_2<0 and dy_2>0 or dx_1>0 and dy_1<0 and dx_2<0 and dy_2<0 or dx_1>0 and dy_1>0 and dx_2>0 and dy_2<0:
                    x_post_2=x_post_2
                    y_post_2=y_post_2
                else:
                    x_post_2=x_ant_2
                    y_post_2=y_ant_2
                dx,dy=Delta_XY_FOV_1_lin(view_1_x,x_post_1+x1_1_crop)
                current_x_new=current_x+dx
                current_y_new=current_y+dy
                dz_1=int(float(Delta_Z_FOV_1_lin(view_1_y,y_post_1+y1_1_crop-70)))
                dx,dy=Delta_XY_FOV_2_lin(view_2_x,x_post_2+x1_2_crop)
                current_x_new=current_x_new+dx
                current_y_new=current_y_new+dy
                dz_2=int(float(Delta_Z_FOV_2_lin(view_2_y,y_post_2+y1_2_crop-70)))
                current_z_new=current_z+int(np.mean([dz_1,dz_2]))
                end=1
        else:
            if x_cen_1==x_post_1:
                print('No posterior detected FOV1')
            else: 
                m = (y_cen_1-y_post_1)/(x_cen_1-x_post_1)
                b = (x_cen_1*y_post_1 - x_post_1*y_cen_1)/(x_cen_1-x_post_1)
                x_ant_1=x_cen_1+(x_cen_1-x_post_1)
                y_ant_1 = m*x_ant_1 + b
                dx_1=(x_cen_1-x_post_1)
                dy_1=(y_cen_1-y_post_1)
                dx_2=(x_cen_2-x_post_2)
                dy_2=(y_cen_2-y_post_2)
                if dx_1<0 and dy_1>0 and dx_2>0 and dy_2>0 or dx_1<0 and dy_1<0 and dx_2<0 and dy_2>0 or dx_1>0 and dy_1<0 and dx_2<0 and dy_2<0 or dx_1>0 and dy_1>0 and dx_2>0 and dy_2<0:
                    x_post_1=x_post_1
                    y_post_1=y_post_1
                else:
                    x_post_1=x_ant_1
                    y_post_1=y_ant_1
                dx,dy=Delta_XY_FOV_1_lin(view_1_x,x_post_1+x1_1_crop)
                current_x_new=current_x+dx
                current_y_new=current_y+dy
                dz_1=int(float(Delta_Z_FOV_1_lin(view_1_y,y_post_1+y1_1_crop-70)))
                dx,dy=Delta_XY_FOV_2_lin(view_2_x,x_post_2+x1_2_crop)
                current_x_new=current_x_new+dx
                current_y_new=current_y_new+dy
                dz_2=int(float(Delta_Z_FOV_2_lin(view_2_y,y_post_2+y1_2_crop-70)))
                current_z_new=current_z+int(np.mean([dz_1,dz_2]))
                end=1
    elif 3 in list_classes_1 and 3 not in list_classes_2 and 2 in list_classes_2 and 1 in list_classes_1 and 1 in list_classes_2:
        print('j')
        list_classes_index_1_post=list_classes_1.index(3)
        list_classes_index_1_cen=list_classes_1.index(1)
        list_classes_index_2_ant=list_classes_2.index(2)
        list_classes_index_2_cen=list_classes_2.index(1)
        x_post_1=xc_rc_1[0][list_classes_index_1_post]
        x_ant_2=xc_rc_2[0][list_classes_index_2_ant]
        y_post_1=yc_rc_1[0][list_classes_index_1_post]
        y_ant_2=yc_rc_2[0][list_classes_index_2_ant]    
        x_cen_1=xc_rc_1[0][list_classes_index_1_cen]
        x_cen_2=xc_rc_2[0][list_classes_index_2_cen]
        y_cen_1=yc_rc_1[0][list_classes_index_1_cen]
        y_cen_2=yc_rc_2[0][list_classes_index_2_cen]
        if x_cen_2==x_ant_2:
            print('No anterior detected FOV2')
        else: 
            m = (y_cen_2-y_ant_2)/(x_cen_2-x_ant_2)
            b = (x_cen_2*y_ant_2 - x_ant_2*y_cen_2)/(x_cen_2-x_ant_2)
            x_post_2=x_cen_2+(x_cen_2-x_ant_2)
            y_post_2 = m*x_post_2 + b
            dx_2=(x_cen_2-x_post_2)
            dy_2=(y_cen_2-y_post_2)
            dx_1=(x_cen_1-x_post_1)
            dy_1=(y_cen_1-y_post_1)
            if dx_1<0 and dy_1>0 and dx_2>0 and dy_2>0 or dx_1<0 and dy_1<0 and dx_2<0 and dy_2>0 or dx_1>0 and dy_1<0 and dx_2<0 and dy_2<0 or dx_1>0 and dy_1>0 and dx_2>0 and dy_2<0:
                x_post_2=x_post_2
                y_post_2=y_post_2
            else:
                x_post_2=x_ant_2
                y_post_2=y_ant_2
            dx,dy=Delta_XY_FOV_1_lin(view_1_x,x_post_1+x1_1_crop)
            current_x_new=current_x+dx
            current_y_new=current_y+dy
            dz_1=int(float(Delta_Z_FOV_1_lin(view_1_y,y_post_1+y1_1_crop-70)))
            dx,dy=Delta_XY_FOV_2_lin(view_2_x,x_post_2+x1_2_crop)
            current_x_new=current_x_new+dx
            current_y_new=current_y_new+dy
            dz_2=int(float(Delta_Z_FOV_2_lin(view_2_y,y_post_2+y1_2_crop-70)))
            current_z_new=current_z+int(np.mean([dz_1,dz_2]))
            end=1

    elif 3 not in list_classes_1 and 3 in list_classes_2 and 2 in list_classes_1 and 1 in list_classes_1 and 1 in list_classes_2:
        print('k')
        list_classes_index_1_ant=list_classes_1.index(2)
        list_classes_index_1_cen=list_classes_1.index(1)
        list_classes_index_2_post=list_classes_2.index(3)
        list_classes_index_2_cen=list_classes_2.index(1)
        x_ant_1=xc_rc_1[0][list_classes_index_1_ant]
        x_post_2=xc_rc_2[0][list_classes_index_2_post]
        y_ant_1=yc_rc_1[0][list_classes_index_1_ant]
        y_post_2=yc_rc_2[0][list_classes_index_2_post]    
        x_cen_1=xc_rc_1[0][list_classes_index_1_cen]
        x_cen_2=xc_rc_2[0][list_classes_index_2_cen]
        y_cen_1=yc_rc_1[0][list_classes_index_1_cen]
        y_cen_2=yc_rc_2[0][list_classes_index_2_cen]
        if x_cen_1==x_ant_1:
            print('No anterior detected FOV1')
        else: 
            m = (y_cen_1-y_ant_1)/(x_cen_1-x_ant_1)
            b = (x_cen_1*y_ant_1 - x_ant_1*y_cen_1)/(x_cen_1-x_ant_1)
            x_post_1=x_cen_1+(x_cen_1-x_ant_1)
            y_post_1 = m*x_post_1 + b
            dx_1=(x_cen_1-x_post_1)
            dy_1=(y_cen_1-y_post_1)
            dx_2=(x_cen_2-x_post_2)
            dy_2=(y_cen_2-y_post_2)
            # add to injection, also neeed new tf, get tf for guessing injection point
            if dx_1<0 and dy_1>0 and dx_2>0 and dy_2>0 or dx_1<0 and dy_1<0 and dx_2<0 and dy_2>0 or dx_1>0 and dy_1<0 and dx_2<0 and dy_2<0 or dx_1>0 and dy_1>0 and dx_2>0 and dy_2<0:
                x_post_1=x_post_1
                y_post_1=y_post_1
            else:
                x_post_1=x_ant_1
                y_post_1=y_ant_1
            dx,dy=Delta_XY_FOV_1_lin(view_1_x,x_post_1+x1_1_crop)
            current_x_new=current_x+dx
            current_y_new=current_y+dy
            dz_1=int(float(Delta_Z_FOV_1_lin(view_1_y,y_post_1+y1_1_crop-70)))
            dx,dy=Delta_XY_FOV_2_lin(view_2_x,x_post_2+x1_2_crop)
            current_x_new=current_x_new+dx
            current_y_new=current_y_new+dy
            dz_2=int(float(Delta_Z_FOV_2_lin(view_2_y,y_post_2+y1_2_crop-70)))
            current_z_new=current_z+int(np.mean([dz_1,dz_2]))
            end=1
    else:
        print('No posterior detected in both FOVs')
    if end==1:
        injection_list_num=1
        XYZ_Location(5000,5000,inj_speed,current_x_new,current_y_new,current_z_new,ser)
        time.sleep(.5)
        # time.sleep(1)
        embryo_coords=function_transformation_matrix_embryo_guess(current_x_new-current_x,current_y_new-current_y,current_z_new-current_z,-37, -44, -25, -8, -26, -35, -41, 3, 10, -16, 1, -21, -22, -31, -14, -8,25,-10,30,40,20,-25,-30,40,-10,15,-20,10)
        x_post_1_new=x_post_1+int(float(embryo_coords.item(0,0)))+9
        y_post_1_new=y_post_1+int(float(embryo_coords.item(1,0)))+24
        x_post_2_new=x_post_2+int(float(embryo_coords.item(2,0)))+8
        y_post_2_new=y_post_2+int(float(embryo_coords.item(3,0)))+11      
        dz=int(np.mean([float(Delta_Z_FOV_1_lin(view_1_y,y_post_1_new+y1_1_crop)),float(Delta_Z_FOV_2_lin(view_2_y,y_post_2_new+y1_2_crop))]))+inj_depth
        print('Injection Depth = ',dz)
        # time.sleep(.15)
        img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,0)
      
        cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/images/image_FOV_1_new_{}.jpg'.format(inj_num),img1)
        cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/images/image_FOV_2_new_{}.jpg'.format(inj_num),img2)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/x_post_1_{}.npy'.format(inj_num),x_post_1)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/y_post_1_{}.npy'.format(inj_num),y_post_1)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/x_post_2_{}.npy'.format(inj_num),x_post_2)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/y_post_2_{}.npy'.format(inj_num),y_post_2)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/x_post_1_new_{}.npy'.format(inj_num),x_post_1_new)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/y_post_1_new_{}.npy'.format(inj_num),y_post_1_new)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/x_post_2_new_{}.npy'.format(inj_num),x_post_2_new)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/y_post_2_new_{}.npy'.format(inj_num),y_post_2_new)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/current_x_{}.npy'.format(inj_num),current_x)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/current_y_{}.npy'.format(inj_num),current_y)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/current_z_{}.npy'.format(inj_num),current_z)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/current_x_new_{}.npy'.format(inj_num),current_x_new)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/current_y_new_{}.npy'.format(inj_num),current_y_new)
        np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/find_posterior_new_data/data/current_z_new_{}.npy'.format(inj_num),current_z_new)
        
        plt.figure(7)
        plt.title('Detected Injection Point FOV1')
        plt.xlabel('x coordinate (px)')
        plt.ylabel('y coordinate (px)')
        plt.plot(x_post_1_new+x1_1_crop,y_post_1_new+y1_1_crop,'ro',markersize=3)
        plt.plot(view_1_x,view_1_y,'bo',markersize=3)
        plt.imshow(img1,cmap='gray')
        plt.figure(8)
        plt.title('Detected Injection Point FOV2')
        plt.xlabel('x coordinate (px)')
        plt.ylabel('y coordinate (px)')
        plt.plot(x_post_2_new+x1_2_crop,y_post_2_new+y1_2_crop,'ro',markersize=3)
        plt.plot(view_2_x,view_2_y,'bo',markersize=3)
        plt.imshow(img2,cmap='gray')
        plt.show()
        current_z_new=current_z_new+dz
        print('Current Z = ',current_z_new) 
        print('Piercing through embryo')
        # change from 2000 to 5000
        XYZ_Location(5000,5000,inj_speed,current_x_new,current_y_new,current_z_new,ser)
        # # Pressure
        # print('Injecting embryo')
        # #comment
        # while q=='x' or q_o=='x' or q_e!='p':
        #     print('Try ',o+1)
        #     signal=continuous_pressure(back_pressure_value_new,pressure_value,'inj')
        #     arduino.write(signal.encode())
        #     q_=arduino.readline()
        #     q_=q_.decode()
        #     print(q_)
        #     q_o=q_[7]
        #     q=q_[8]
        #     q_e=q_[len(q_)-3]
        #     o+=1
        # #comment
        time.sleep(pressure_time)
        #comment
        #new
        current_z=current_z-inj_depth-300
        # Come out
        #comment
        XYZ_Location(20000,20000,inj_speed,current_x_new,current_y_new,current_z_new,ser)
        #comment
        # #new
        # print('Pressure done')
        # if pressure_time==1 or pressure_time==2:
        #     time.sleep(1)
        # while back_pressure_value_new>back_pressure_value:
        #     back_pressure_value_new=pressure_value-(press_count)
        #     print(back_pressure_value_new)
        #     l='x'
        #     l_o='x'
        #     l_e='1'
        #     k=0
        #     while l=='x' or l_o=='x' or l_e!='p':
        #         print('Try ',k+1)
        #         signal=continuous_pressure(back_pressure_value_new,pressure_value,'bp')
        #         arduino.write(signal.encode())
        #         l_=arduino.readline()
        #         l_=l_.decode()
        #         print(l_)
        #         l_o=l_[7]
        #         l=l_[8]
        #         l_e=l_[len(l_)-3]
        #         k+=1
        #     press_count+=1
    else:
        print('Second Attempt Failed')
    return current_x_new,current_y_new,current_z_new,injection_list_num

# import tensorflow as tf 

# graph = tf.Graph()
# with graph.as_default():
#   od_graph_def = tf.compat.v1.GraphDef()
#   with tf.compat.v2.io.gfile.GFile('C:/Users/User/Downloads/Andrew_files/faster_r_cnn_trained_model_injection_point_tip_pipette_robot_2_new_4'+'/frozen_inference_graph.pb', 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
# with graph.as_default():
#     with tf.compat.v1.Session() as sess:
#         current_x,current_y,current_z,x_post_1,y_post_1,x_post_2,y_post_2,injection_list_num=find_posterior_new(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,600,800,507,800,graph,sess)

# img1=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/Detect_injection/post_fov_1_531.jpg',1)
# img2=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/Detect_injection/post_fov_2_531.jpg',1)
# plt.figure(1)
# plt.title('Detected Points')
# plt.xlabel('x coordinate (px)')
# plt.ylabel('y coordinate (px)')
# plt.plot(x_post_1,y_post_1,'ro',markersize=4)
# plt.imshow(img1, cmap='gray')
# plt.show()

# plt.figure(2)
# plt.title('Detected Points')
# plt.xlabel('x coordinate (px)')
# plt.ylabel('y coordinate (px)')
# plt.plot(x_post_2,y_post_2,'ro',markersize=4)
# plt.imshow(img2, cmap='gray')
# plt.show()
