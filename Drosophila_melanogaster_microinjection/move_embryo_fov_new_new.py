# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 08:46:37 2021

@author: User
"""

from XYZ_Stage.XYZ_Position import XYZ_Location
from Delta_XY_FOV_1_lin import Delta_XY_FOV_1_lin
from Delta_XY_FOV_2_lin import Delta_XY_FOV_2_lin
from Delta_Z_FOV_1_lin import Delta_Z_FOV_1_lin
from Delta_Z_FOV_2_lin import Delta_Z_FOV_2_lin
from ML.ml_injection_point_estimation_new import ml_injection_point_estimation_new
import time
from matplotlib import pyplot as plt
from Pressure_Control.Continuous_Pressure import continuous_pressure
from stream_image import stream_image
import numpy as np
from math import pi,cos,sin
import cv2

def move_embryo_fov_new_new(fov,action,X_est,Y_est,Z_est,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,current_x,current_y,current_z,view_1_x,view_1_y,view_2_x,view_2_y,footage_socket_1,footage_socket_2,inj_num,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_height_1,im_width_1,im_height_2,im_width_2,graph,sess,ser,time_wait,next_z,post_z,current_z_new,move_num,pic):
    
    end=0
    injection_list_num=0
    x_coord_emb_1=0
    y_coord_emb_1=0
    x_coord_tip_1=view_1_x
    y_coord_tip_1=view_1_y
    x_coord_emb_2=0
    y_coord_emb_2=0
    x_coord_tip_2=view_2_x
    y_coord_tip_2=view_2_y
    x_cen_1=0
    y_cen_1=0
    x_cen_2=0
    y_cen_2=0
    x_post_1=0
    y_post_1=0
    x_post_2=0
    y_post_2=0
    q='x'
    q_o='x'
    q_e='1'
    press_count=1
    o=0
    back_pressure_value_new=pressure_value

    if fov==1 and action=='centroid':
        current_z_new=current_z
        # FOV 1 Center
        time.sleep(time_wait)
        img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,1)
        img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop]    
        output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc,y2a_rc,x1a_rc,x2a_rc,xc_rc,yc_rc=ml_injection_point_estimation_new([img1_crop],.01,im_height_1,im_width_1,graph,sess,1)
        list_classes=output_dict_detection_classes_stored[0].tolist()
        if 1 not in list_classes:
            print('No centroid detected FOV1')
            current_x=X_est
            current_y=Y_est
            current_z=Z_est
            end=2
        else:
            list_classes_index=list_classes.index(1)
            plt.figure(1)
            plt.title('Detected Centroid FOV1')
            plt.xlabel('x coordinate (px)')
            plt.ylabel('y coordinate (px)')
            plt.plot(xc_rc[0][list_classes_index]+x1_1_crop,yc_rc[0][list_classes_index]+y1_1_crop,'ro',markersize=3)
            plt.plot(view_1_x,view_1_y,'bo',markersize=3)
            plt.imshow(img1,cmap='gray')
            plt.show()
            dx,dy=Delta_XY_FOV_1_lin(view_1_x,xc_rc[0][list_classes_index]+x1_1_crop)
            dx=dx
            dy=dy
            print('Move up half z for FOV_1')
            dz=int(float(Delta_Z_FOV_1_lin(view_1_y,yc_rc[0][list_classes_index]+y1_1_crop-105)))
            current_z=current_z+dz
            print('Current Z = ',current_z)     
            print('Moving embryo center under needle FOV1')
            XYZ_Location(5000,5000,2000,current_x,current_y,current_z,ser)
            time.sleep(.25)
            current_x=current_x+dx
            current_y=current_y+dy
            x_coord_emb_1=xc_rc[0][list_classes_index]+x1_1_crop
            y_coord_emb_1=yc_rc[0][list_classes_index]+y1_1_crop
            x_cen_1=x_coord_emb_1
            y_cen_1=y_coord_emb_1
            x_coord_tip_1=view_1_x
            y_coord_tip_1=view_1_y
            print('Current X = ',current_x)   
            print('Current Y = ',current_y)
            XYZ_Location(5000,5000,2000,current_x,current_y,current_z,ser)
            time.sleep(.25)
        next_z=current_z
    elif fov==2 and action=='centroid':
        current_z_new=current_z
        # time.sleep(.15)
        img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,1)
        img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop]    
        output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc,y2a_rc,x1a_rc,x2a_rc,xc_rc,yc_rc=ml_injection_point_estimation_new([img2_crop],.01,im_height_2,im_width_2,graph,sess,2)
        list_classes=output_dict_detection_classes_stored[0].tolist()
        if 1 not in list_classes:
            print('No centroid detected FOV2')
            current_x=current_x
            current_y=current_y
            current_z=current_z
            end=1
            print('Done centroid FOV2')
        else:
            list_classes_index=list_classes.index(1)
            plt.figure(2)
            plt.title('Detected Centroid FOV2')
            plt.xlabel('x coordinate (px)')
            plt.ylabel('y coordinate (px)')
            plt.plot(xc_rc[0][list_classes_index]+x1_2_crop,yc_rc[0][list_classes_index]+y1_2_crop,'ro',markersize=3)
            plt.plot(view_2_x,view_2_y,'bo',markersize=3)
            plt.imshow(img2,cmap='gray')
            plt.show()
            dx,dy=Delta_XY_FOV_2_lin(view_2_x,xc_rc[0][list_classes_index]+x1_2_crop)
            print('Move up half z for FOV_2')
            dz=int(float(Delta_Z_FOV_2_lin(view_2_y,yc_rc[0][list_classes_index]+y1_2_crop-90)))
            current_z=current_z+dz
            print('Current Z = ',current_z)   
            print('Moving embryo center under needle FOV2')
            XYZ_Location(5000,5000,2000,current_x,current_y,current_z,ser)
            time.sleep(.25)
            current_x=current_x+dx
            current_y=current_y+dy
            x_coord_emb_2=xc_rc[0][list_classes_index]+x1_2_crop
            y_coord_emb_2=yc_rc[0][list_classes_index]+y1_2_crop
            x_cen_2=x_coord_emb_2
            y_cen_2=y_coord_emb_2
            x_coord_tip_2=view_2_x
            y_coord_tip_2=view_2_y
            print('Current X = ',current_x)   
            print('Current Y = ',current_y)   
            XYZ_Location(5000,5000,2000,current_x,current_y,current_z,ser)
            time.sleep(.25)
        next_z=current_z
    elif action=='no move':
            next_z=next_z
            # time.sleep(.15)
            img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,0)
            img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop]    
            output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc,y2a_rc,x1a_rc,x2a_rc,xc_rc,yc_rc=ml_injection_point_estimation_new([img1_crop],.0001,im_height_1,im_width_1,graph,sess,3)
            list_classes=output_dict_detection_classes_stored[0].tolist()
            current_z_new=current_z
            if 3 in list_classes and 1 in list_classes:
                print('Posterior detected FOV1')
                list_classes_index=list_classes.index(3)
                list_classes_index_cen=list_classes.index(1)
                plt.figure(3)
                plt.title('Detected Injection Point FOV1')
                plt.xlabel('x coordinate (px)')
                plt.ylabel('y coordinate (px)')
                plt.plot(xc_rc[0][list_classes_index]+x1_1_crop,yc_rc[0][list_classes_index]+y1_1_crop,'ro',markersize=3)
                plt.plot(view_1_x,view_1_y,'bo',markersize=3)
                plt.imshow(img1,cmap='gray')
                plt.show()
                dx,dy=Delta_XY_FOV_1_lin(view_1_x,xc_rc[0][list_classes_index]+x1_1_crop)
                current_x_new=current_x+dx
                current_y_new=current_y+dy
                dz_1=int(float(Delta_Z_FOV_1_lin(view_1_y,yc_rc[0][list_classes_index]+y1_1_crop-70)))
                x_coord_emb_1=xc_rc[0][list_classes_index]+x1_1_crop
                y_coord_emb_1=yc_rc[0][list_classes_index]+y1_1_crop
                x_cen_1=xc_rc[0][list_classes_index_cen]+x1_1_crop
                y_cen_1=yc_rc[0][list_classes_index_cen]+y1_1_crop
                x_coord_tip_1=view_1_x
                y_coord_tip_1=view_1_y
            else:
                print('Rotating image')
                center=(int((float(im_width_1))/(2)),int((float(im_height_1))/(2)))
                scale=1
                M_1 = cv2.getRotationMatrix2D(center,90, scale)
                cosine = np.abs(M_1[0, 0])
                sine = np.abs(M_1[0, 1])
                nW = int((im_height_1 * sine) + (im_width_1 * cosine))
                nH = int((im_height_1 * cosine) + (im_width_1 * sine))
                M_1[0, 2] += (nW / 2) - int((float(im_width_1))/(2))
                M_1[1, 2] += (nH / 2) - int((float(im_height_1))/(2))
                img1_crop=cv2.warpAffine(img1_crop, M_1, (im_height_1, im_width_1)) 
                img1_crop=cv2.resize(img1_crop,(800,600))
                output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc,y2a_rc,x1a_rc,x2a_rc,xc_rc,yc_rc=ml_injection_point_estimation_new([img1_crop],.0001,600,800,graph,sess,3)
                list_classes=output_dict_detection_classes_stored[0].tolist()
                if 3 in list_classes and 1 in list_classes:
                    print('Posterior detected FOV1')
                    list_classes_index=list_classes.index(3)
                    list_classes_index_cen=list_classes.index(1)
                    x_new=(xc_rc[0][list_classes_index])*(im_height_1/800)
                    y_new=(yc_rc[0][list_classes_index])*(im_width_1/600)
                    xm=im_height_1/2
                    ym=im_width_1/2
                    a=-90*(pi/180)
                    xc_rotate=(y_new-ym)*sin(a)+(x_new-xm)*cos(a)+ym
                    yc_rotate=(y_new-ym)*cos(a)-(x_new-xm)*sin(a)+xm
                    plt.figure(3)
                    plt.title('Detected Injection Point FOV1')
                    plt.xlabel('x coordinate (px)')
                    plt.ylabel('y coordinate (px)')
                    plt.plot(xc_rotate+x1_1_crop,yc_rotate+y1_1_crop,'ro',markersize=3)
                    plt.plot(view_1_x,view_1_y,'bo',markersize=3)
                    plt.imshow(img1,cmap='gray')
                    plt.show()
                    dx,dy=Delta_XY_FOV_1_lin(view_1_x,xc_rotate+x1_1_crop)
                    current_x_new=current_x+dx
                    current_y_new=current_y+dy
                    dz_1=int(float(Delta_Z_FOV_1_lin(view_1_y,yc_rotate+y1_1_crop-70)))
                    x_coord_emb_1=xc_rotate+x1_1_crop
                    y_coord_emb_1=yc_rotate+y1_1_crop
                    x_cen_1=xc_rc[0][list_classes_index_cen]+x1_1_crop
                    y_cen_1=yc_rc[0][list_classes_index_cen]+y1_1_crop
                    x_coord_tip_1=view_1_x
                    y_coord_tip_1=view_1_y
                else:
                    end=1
                    print('Done posterior FOV1')
            img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop]
            output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc,y2a_rc,x1a_rc,x2a_rc,xc_rc,yc_rc=ml_injection_point_estimation_new([img2_crop],.0001,im_height_2,im_width_2,graph,sess,6)
            list_classes=output_dict_detection_classes_stored[0].tolist()
            if 3 in list_classes and 1 in list_classes and end!=1:
                print('Posterior detected FOV2')
                list_classes_index_2=list_classes.index(3)
                list_classes_index_cen_2=list_classes.index(1)
                plt.figure(6)
                plt.title('Detected Injection Point FOV2')
                plt.xlabel('x coordinate (px)')
                plt.ylabel('y coordinate (px)')
                plt.plot(xc_rc[0][list_classes_index_2]+x1_2_crop,yc_rc[0][list_classes_index_2]+y1_2_crop,'ro',markersize=3)
                plt.plot(view_2_x,view_2_y,'bo',markersize=3)
                plt.imshow(img2,cmap='gray')
                plt.show()
                dx,dy=Delta_XY_FOV_2_lin(view_2_x,xc_rc[0][list_classes_index_2]+x1_2_crop)
                current_x_new=current_x_new+dx
                current_y_new=current_y_new+dy
                dz_2=int(float(Delta_Z_FOV_2_lin(view_2_y,yc_rc[0][list_classes_index_2]+y1_2_crop-70)))
                current_z_new=current_z+int(np.mean([dz_1,dz_2]))       
                x_coord_emb_2=xc_rc[0][list_classes_index_2]+x1_2_crop
                y_coord_emb_2=yc_rc[0][list_classes_index_2]+y1_2_crop
                x_cen_2=xc_rc[0][list_classes_index_cen_2]+x1_2_crop
                y_cen_2=yc_rc[0][list_classes_index_cen_2]+y1_2_crop
                x_coord_tip_2=view_2_x
                y_coord_tip_2=view_2_y
                dx_2=(x_cen_2-x_coord_emb_2)
                dy_2=(y_cen_2-y_coord_emb_2)
                dx_1=(x_cen_1-x_coord_emb_1)
                dy_1=(y_cen_1-y_coord_emb_1)
                if abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1<0 and dy_1>0 and dx_2>0 and dy_2>0 or abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1<0 and dy_1<0 and dx_2<0 and dy_2>0 or abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1>0 and dy_1<0 and dx_2<0 and dy_2<0 or abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1>0 and dy_1>0 and dx_2>0 and dy_2<0:
                    current_x=current_x
                    current_y=current_y
                else:
                    current_x=current_x_new
                    current_y=current_y_new
            else:
                print('Rotating image')
                center=(int((float(im_width_2))/(2)),int((float(im_height_2))/(2)))
                scale=1
                M_1 = cv2.getRotationMatrix2D(center,90, scale)
                cosine = np.abs(M_1[0, 0])
                sine = np.abs(M_1[0, 1])
                nW = int((im_height_2 * sine) + (im_width_2 * cosine))
                nH = int((im_height_2 * cosine) + (im_width_2 * sine))
                M_1[0, 2] += (nW / 2) - int((float(im_width_2))/(2))
                M_1[1, 2] += (nH / 2) - int((float(im_height_2))/(2))
                img2_crop=cv2.warpAffine(img2_crop, M_1, (im_height_2, im_width_2)) 
                img2_crop=cv2.resize(img2_crop,(800,600))
                output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc,y2a_rc,x1a_rc,x2a_rc,xc_rc,yc_rc=ml_injection_point_estimation_new([img2_crop],.0001,600,800,graph,sess,6)
                list_classes=output_dict_detection_classes_stored[0].tolist()
                if 3 in list_classes and 1 in list_classes and end!=1:
                    print('Posterior detected FOV2')
                    list_classes_index_2=list_classes.index(3)
                    list_classes_index_cen_2=list_classes.index(1)
                    x_new=(xc_rc[0][list_classes_index_2])*(im_height_2/800)
                    y_new=(yc_rc[0][list_classes_index_2])*(im_width_2/600)
                    xm=im_height_2/2
                    ym=im_width_2/2
                    a=-90*(pi/180)
                    xc_rotate=(y_new-ym)*sin(a)+(x_new-xm)*cos(a)+ym
                    yc_rotate=(y_new-ym)*cos(a)-(x_new-xm)*sin(a)+xm
                    plt.figure(6)
                    plt.title('Detected Injection Point FOV2')
                    plt.xlabel('x coordinate (px)')
                    plt.ylabel('y coordinate (px)')
                    plt.plot(xc_rotate+x1_2_crop,yc_rotate+y1_2_crop,'ro',markersize=3)
                    plt.plot(view_2_x,view_2_y,'bo',markersize=3)
                    plt.imshow(img2,cmap='gray')
                    plt.show()
                    dx,dy=Delta_XY_FOV_2_lin(view_2_x,xc_rotate+x1_2_crop)
                    current_x_new=current_x_new+dx
                    current_y_new=current_y_new+dy
                    dz_2=int(float(Delta_Z_FOV_2_lin(view_2_y,yc_rotate+y1_2_crop-70)))
                    current_z_new=current_z+int(np.mean([dz_1,dz_2]))       
                    x_coord_emb_2=xc_rotate+x1_2_crop
                    y_coord_emb_2=yc_rotate+y1_2_crop
                    x_cen_2=xc_rc[0][list_classes_index_cen_2]+x1_2_crop
                    y_cen_2=yc_rc[0][list_classes_index_cen_2]+y1_2_crop
                    x_coord_tip_2=view_2_x
                    y_coord_tip_2=view_2_y
                    dx_2=(x_cen_2-x_coord_emb_2)
                    dy_2=(y_cen_2-y_coord_emb_2)
                    dx_1=(x_cen_1-x_coord_emb_1)
                    dy_1=(y_cen_1-y_coord_emb_1)
                    if abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1<0 and dy_1>0 and dx_2>0 and dy_2>0 or abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1<0 and dy_1<0 and dx_2<0 and dy_2>0 or abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1>0 and dy_1<0 and dx_2<0 and dy_2<0 or abs(x_coord_emb_1-x_coord_tip_1)<=5*move_num and abs(x_coord_emb_2-x_coord_tip_2)<=5*move_num and dx_1>0 and dy_1>0 and dx_2>0 and dy_2<0:
                        current_x=current_x
                        current_y=current_y
                    else:
                        current_x=current_x_new
                        current_y=current_y_new
                else:
                    print('Done posterior FOV2')
                    end=1
    elif action=='just move':
        next_z=next_z
        print('Current X = ',current_x)   
        print('Current Y = ',current_y) 
        print('Current Z = ',current_z)
        print('Moving embryo injecion point under needle')
        XYZ_Location(5000,5000,2000,current_x,current_y,current_z,ser)
        # time.sleep(.25)
        time.sleep(.5)   
    elif action=='inject new':
        rot_1=0
        rot_2=0
        next_z=next_z
        current_z_new=current_z
        # time.sleep(.15)
        img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,0)
        img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop]    
        output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc_1,y2a_rc_1,x1a_rc_1,x2a_rc_1,xc_rc_1,yc_rc_1=ml_injection_point_estimation_new([img1_crop],.0001,im_height_1,im_width_1,graph,sess,3)
        list_classes_1=output_dict_detection_classes_stored[0].tolist()
        if 3 not in list_classes_1:
            print('Rotating image FOV 1')
            center=(int((float(im_width_1))/(2)),int((float(im_height_1))/(2)))
            scale=1
            M_1 = cv2.getRotationMatrix2D(center,90, scale)
            cosine = np.abs(M_1[0, 0])
            sine = np.abs(M_1[0, 1])
            nW = int((im_height_1 * sine) + (im_width_1 * cosine))
            nH = int((im_height_1 * cosine) + (im_width_1 * sine))
            M_1[0, 2] += (nW / 2) - int((float(im_width_1))/(2))
            M_1[1, 2] += (nH / 2) - int((float(im_height_1))/(2))
            img1_crop=cv2.warpAffine(img1_crop, M_1, (im_height_1, im_width_1)) 
            img1_crop=cv2.resize(img1_crop,(800,600))
            output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc_1,y2a_rc_1,x1a_rc_1,x2a_rc_1,xc_rc_1,yc_rc_1=ml_injection_point_estimation_new([img1_crop],.0001,600,800,graph,sess,3)
            list_classes_1=output_dict_detection_classes_stored[0].tolist()
            if 3 in list_classes_1:
                rot_1=1
                print('Rotated FOV 1 correctly')
        img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop] 
        output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc_2,y2a_rc_2,x1a_rc_2,x2a_rc_2,xc_rc_2,yc_rc_2=ml_injection_point_estimation_new([img2_crop],.0001,im_height_2,im_width_2,graph,sess,2)
        list_classes_2=output_dict_detection_classes_stored[0].tolist()
        if 3 not in list_classes_2:
            print('Rotating image FOV 2')
            center=(int((float(im_width_2))/(2)),int((float(im_height_2))/(2)))
            scale=1
            M_2 = cv2.getRotationMatrix2D(center,90, scale)
            cosine = np.abs(M_2[0, 0])
            sine = np.abs(M_2[0, 1])
            nW = int((im_height_2 * sine) + (im_width_2 * cosine))
            nH = int((im_height_2 * cosine) + (im_width_2 * sine))
            M_2[0, 2] += (nW / 2) - int((float(im_width_2))/(2))
            M_2[1, 2] += (nH / 2) - int((float(im_height_2))/(2))
            img2_crop=cv2.warpAffine(img2_crop, M_2, (im_height_2, im_width_2)) 
            img2_crop=cv2.resize(img2_crop,(800,600))
            output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,y1a_rc_2,y2a_rc_2,x1a_rc_2,x2a_rc_2,xc_rc_2,yc_rc_2=ml_injection_point_estimation_new([img2_crop],.0001,600,800,graph,sess,3)
            list_classes_2=output_dict_detection_classes_stored[0].tolist()
            if 3 in list_classes_2:
                rot_2=1
                print('Rotated FOV 2 correctly')
        if 3 in list_classes_1 and 1 in list_classes_1 and rot_1==0:
            print('Posterior detected FOV1')
            list_classes_index_1=list_classes_1.index(3)
            list_classes_index_cen_1=list_classes_1.index(1)
            x_post_1=xc_rc_1[0][list_classes_index_1]
            y_post_1=yc_rc_1[0][list_classes_index_1]
            x_cen_1=xc_rc_1[0][list_classes_index_cen_1]
            y_cen_1=yc_rc_1[0][list_classes_index_cen_1]
        elif 3 in list_classes_1 and 1 in list_classes_1 and rot_1==1:
            print('Posterior detected FOV1')
            list_classes_index_1=list_classes_1.index(3)
            list_classes_index_cen_1=list_classes_1.index(1)
            x_new=(xc_rc_1[0][list_classes_index_1])*(im_height_1/800)
            y_new=(yc_rc_1[0][list_classes_index_1])*(im_width_1/600)
            x_cen_1=(xc_rc_1[0][list_classes_index_cen_1])*(im_height_1/800)
            y_cen_1=(yc_rc_1[0][list_classes_index_cen_1])*(im_width_1/600)
            xm=im_height_1/2
            ym=im_width_1/2
            a=-90*(pi/180)
            x_post_1=(y_new-ym)*sin(a)+(x_new-xm)*cos(a)+ym
            y_post_1=(y_new-ym)*cos(a)-(x_new-xm)*sin(a)+xm
            x_cen_1=(y_cen_1-ym)*sin(a)+(x_cen_1-xm)*cos(a)+ym
            y_cen_1=(y_cen_1-ym)*cos(a)-(x_cen_1-xm)*sin(a)+xm
        else:
            print('Done')
            end=1
        if 3 in list_classes_2 and 1 in list_classes_2 and end!=1 and rot_2==0:
            print('Posterior detected FOV2')            
            list_classes_index_2=list_classes_2.index(3)
            list_classes_index_cen_2=list_classes_2.index(1)
            x_post_2=xc_rc_2[0][list_classes_index_2]
            y_post_2=yc_rc_2[0][list_classes_index_2]
            x_cen_2=xc_rc_2[0][list_classes_index_cen_2]
            y_cen_2=yc_rc_2[0][list_classes_index_cen_2]
        elif 3 in list_classes_2 and 1 in list_classes_2 and end!=1 and rot_2==1:
            print('Posterior detected FOV2')
            list_classes_index_2=list_classes_2.index(3)
            list_classes_index_cen_2=list_classes_2.index(1)
            x_new=(xc_rc_2[0][list_classes_index_2])*(im_height_2/800)
            y_new=(yc_rc_2[0][list_classes_index_2])*(im_width_2/600)
            x_cen_2=(xc_rc_2[0][list_classes_index_cen_2])*(im_height_1/800)
            y_cen_2=(yc_rc_2[0][list_classes_index_cen_2])*(im_width_1/600)
            xm=im_height_2/2
            ym=im_width_2/2
            a=-90*(pi/180)
            x_post_2=(y_new-ym)*sin(a)+(x_new-xm)*cos(a)+ym
            y_post_2=(y_new-ym)*cos(a)-(x_new-xm)*sin(a)+xm
            x_cen_2=(y_cen_2-ym)*sin(a)+(x_cen_2-xm)*cos(a)+ym
            y_cen_2=(y_cen_2-ym)*cos(a)-(x_cen_2-xm)*sin(a)+xm
        else:
            print('Done')
            end=1
        dx_2=(x_cen_2-x_post_2)
        dy_2=(y_cen_2-y_post_2)
        dx_1=(x_cen_1-x_post_1)
        dy_1=(y_cen_1-y_post_1)
        # if 3 in list_classes_2 and 1 in list_classes_2 and end!=1 and dx_1<0 and dy_1>0 and dx_2>0 and dy_2>0 or dx_1<0 and dy_1<0 and dx_2<0 and dy_2>0 or dx_1>0 and dy_1<0 and dx_2<0 and dy_2<0 or dx_1>0 and dy_1>0 and dx_2>0 and dy_2<0:
        if end!=1:
            cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Surface_injection/images/post_fov_1_{}.jpg'.format(inj_num),img1_crop)
            cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Surface_injection/images/post_fov_2_{}.jpg'.format(inj_num),img2_crop)
            np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Surface_injection/data/post_fov_1_x_{}.npy'.format(inj_num),x_post_1)
            np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Surface_injection/data/post_fov_1_y_{}.npy'.format(inj_num),y_post_1)
            np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Surface_injection/data/post_fov_2_x_{}.npy'.format(inj_num),x_post_2)
            np.save('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Surface_injection/data/post_fov_2_y_{}.npy'.format(inj_num),y_post_2)
            injection_list_num=1
            dz=int(np.mean([float(Delta_Z_FOV_1_lin(view_1_y,y_post_1+y1_1_crop)),float(Delta_Z_FOV_2_lin(view_2_y,y_post_2+y1_2_crop))]))+inj_depth
            print('Injection Depth = ',dz)
            plt.figure(7)
            plt.title('Detected Injection Point FOV1')
            plt.xlabel('x coordinate (px)')
            plt.ylabel('y coordinate (px)')
            plt.plot(x_post_1+x1_1_crop,y_post_1+y1_1_crop,'ro',markersize=3)
            plt.plot(view_1_x,view_1_y,'bo',markersize=3)
            plt.imshow(img1,cmap='gray')
            plt.figure(8)
            plt.title('Detected Injection Point FOV2')
            plt.xlabel('x coordinate (px)')
            plt.ylabel('y coordinate (px)')
            plt.plot(x_post_2+x1_2_crop,y_post_2+y1_2_crop,'ro',markersize=3)
            plt.plot(view_2_x,view_2_y,'bo',markersize=3)
            plt.imshow(img2,cmap='gray')
            plt.show()
            current_z=current_z+dz
            print('Current Z = ',current_z) 
            print('Piercing through embryo')
            # change from 2000 to 5000
            XYZ_Location(5000,5000,inj_speed,current_x,current_y,current_z,ser)
            # Pressure
            print('Injecting embryo')
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
            XYZ_Location(20000,20000,inj_speed,current_x,current_y,current_z,ser)
            time.sleep(1.5)
            img1,img2=stream_image(footage_socket_1,footage_socket_2,pic,0)
            img1_crop=img1[y1_1_crop:y2_1_crop,x1_1_crop:x2_1_crop] 
            img2_crop=img2[y1_2_crop:y2_2_crop,x1_2_crop:x2_2_crop] 
            # #comment
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
            #     # press_count+=2
            # #comment
            cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Detect_injection/post_fov_1_{}.jpg'.format(inj_num),img1_crop)
            cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Detect_injection/post_fov_2_{}.jpg'.format(inj_num),img2_crop)
            end=1
        else:
            print('no injection')
            end=1
    else:
        print('Doing nothing')

    return end,current_x,current_y,current_z,next_z,injection_list_num,x_coord_emb_1,y_coord_emb_1,x_coord_tip_1,y_coord_tip_1,x_coord_emb_2,y_coord_emb_2,x_coord_tip_2,y_coord_tip_2,arduino,y1_1_crop,y2_1_crop,x1_1_crop,x2_1_crop,y1_2_crop,y2_2_crop,x1_2_crop,x2_2_crop,im_width_1,im_height_1,im_width_2,im_height_2,current_z_new,move_num
