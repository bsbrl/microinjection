# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:16:14 2022

@author: User
"""


import time
from XYZ_Stage.XYZ_Position import XYZ_Location
from injection_ml_tip_short_centroid_new import injection_ml_tip_short_centroid_new

def injection_results(elim_embryo,filename,time_wait,view_1_x,view_1_y,view_2_x,view_2_y,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,current_x_centroid,current_y_centroid,current_z_centroid,total_embryos):
    print('Turn off stream, watch video, and then turn on stream again')
    non_inj=int(input('How many embryos did not get injected? '))
    e_num_list=[]
    for i in range(non_inj):
        e_num=int(input('Embryo number =  '))
        e_num_list.append(e_num)
    for i in e_num_list:
        XYZ_Location(20000,20000,8000,elim_embryo[i-1][0],elim_embryo[i-1][1],15000,ser)
        time.sleep(4)
        dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,arduino,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid=injection_ml_tip_short_centroid_new(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,elim_embryo[i-1][0],elim_embryo[i-1][1],elim_embryo[i-1][2],dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,i-1)    
        XYZ_Location(20000,20000,8000,current_x_centroid,current_y_centroid,current_z_centroid,ser)
        time.sleep(3)
        XYZ_Location(20000,20000,8000,current_x_centroid,current_y_centroid,current_z_centroid+500,ser)
        time.sleep(3)
        XYZ_Location(20000,20000,8000,current_x_centroid,current_y_centroid,current_z_centroid,ser)
        time.sleep(3)
        print('X = ',current_x_centroid)
        print('Y = ',current_y_centroid)
        print('Z = ',current_z_centroid)
    # Bring back to 0
    XYZ_Location(10000,10000,8000,current_x_centroid,current_y_centroid,0,ser)
    print(filename[7:21])
    print('{} % of dish injected'.format(float(float(len(elim_embryo)-non_inj)/float(len(elim_embryo)))*100))
    print('Number of embryos injected = ',total_embryos-len(elim_embryo))

# img_dish=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/Entire_Petri_Dish_710.jpg',1)
# img_dish_new=cv2.resize(img_dish,(1600,1067))
# img_positions=[[1968,964,2010,993],[2188,1013,2238,1039],[1985,1113,2011,1158]]
# injection_results(img_dish_new,img_positions)
    