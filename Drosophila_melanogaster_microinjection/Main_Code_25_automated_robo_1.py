# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:18:59 2022

@author: User
"""

from XYZ_Stage.XYZ_Position import XYZ_Location
from DSLR_Camera.DSLR_Call import func_TakeNikonPicture
from ML.ml_whole_image import ml
from ML.rowcol_fun import rowcol_fun
from ML.order import order 
import time
import cv2
import numpy as np
from injection_ml_tip_short_new_thresh import injection_ml_tip_short_new_thresh
from new_pipette_new import new_pipette_new
from first_pipette import first_pipette
from most_optimal_path import most_optimal_path
from ML.detections_dslr_image import detections_dslr_image
import tensorflow as tf
import math
import serial
from ML.transformation_matrix_DSLR_pipette import function_transformation_matrix_DSLR_pipette
from new_z import new_z
import zmq
from injection_results_new import injection_results_new

total_start_time=time.time()
# Initial Variables
z_needle=20000
# z_needle=23000
# z_needle=19000
Z_initial=z_needle+5000      
width_image=6000
height_image=4000
thresh_ml=.1
sum_image_thresh_max=20000
sum_image_thresh_min=3000
target_pixel=6000
over_injected=0
under_injected=0
missed=0
# inj_num=int(input('inj_num = '))
inj_num=196
over_injected=0
under_injected=0
missed=0
no_injected=0
set_0=0
switch=0
inj_range=0
inj_num_init=inj_num
pressure_value=1
back_pressure_value=10
pressure_time=4
inj_depth=-15
# inj_depth=0
post_z=-200
# inj_speed=2000
inj_speed=1000
pipette=1
calib_pipette_num=1
pip_num=0
num=0
correct=0
pip_em_num=[0]
switch_list=[]
injected_embryos=0
injected_embryos_count=0
injected=2
V=math.hypot(20000,20000)
inv_V=(float(float(1)/float(V)))
injection_list=[]
injection_list_num_list=[]
injected_0=[]
injected_list=[]
y1a_rc_new=[]
y2a_rc_new=[]
x1a_rc_new=[]
x2a_rc_new=[]
x1a_rc_post_new=[]
y1a_rc_post_new=[]
x2a_rc_post_new=[]
y2a_rc_post_new=[]
elim_embryo=[]
deltas_pipette=[[0,0,0]]
injection_time_list=[]

# Open sockets
context = zmq.Context()
footage_socket_1 = context.socket(zmq.SUB)
footage_socket_1.bind('tcp://*:5555')
footage_socket_1.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
footage_socket_2 = context.socket(zmq.SUB)
footage_socket_2.bind('tcp://*:4555')
footage_socket_2.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
# Connect XYZ stage
ser = serial.Serial('COM3', 9600,timeout = 5)
if not ser.isOpen():
    ser.open()
# Connect to arduino
arduino = serial.Serial('COM7', 9600, timeout = 5)
if not arduino.isOpen():
    arduino.open()
time.sleep(5)
print('Connecting to arduino')

# Go to camera 
print('Moving under DSLR')
# take picture
filename='Entire_Petri_Dish_991.jpg'
XYZ_Location(10000,10000,8000,54430,93000,5000,ser)
time.sleep(15)
func_TakeNikonPicture(filename)
time.sleep(10)
image=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename)
image_height=6000
image_width=4000
center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
scale=1
fromCenter=False
M_1 = cv2.getRotationMatrix2D(center,270, scale)
cosine = np.abs(M_1[0, 0])
sine = np.abs(M_1[0, 1])
nW = int((image_height * sine) + (image_width * cosine))
nH = int((image_height * cosine) + (image_width * sine))
M_1[0, 2] += (nW / 2) - int((float(image_width))/(2))
M_1[1, 2] += (nH / 2) - int((float(image_height))/(2))
new_1=cv2.warpAffine(image, M_1, (image_height, image_width)) 
cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,new_1)

# detect embryos
print('Detecting embryos')
xc_rc,yc_rc,scores=ml('C:/Users/User/Downloads/Andrew_files/faster_r_cnn_trained_model_petri_new_8','C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,thresh_ml,width_image,height_image)
img_dish=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
img_dish_new=cv2.resize(img_dish,(1600,1067))
y1a_rc,y2a_rc,x1a_rc,x2a_rc=order(xc_rc,yc_rc,0,0)
print(len(xc_rc))
print(len(y1a_rc))
for i in range(len(y1a_rc)):
    cv2.rectangle(img_dish_new,(int(x1a_rc[i][0]*(float(1600)/float(6000))),int(y1a_rc[i][0]*(float(1067)/float(4000)))),(int(x2a_rc[i][0]*(float(1600)/float(6000))),int(y2a_rc[i][0]*(float(1067)/float(4000)))),(0,255,0),1)
cv2.rectangle(img_dish_new,(int(0*(float(1600)/float(6000))),int(0*(float(1067)/float(4000)))),(int(4080*(float(1600)/float(6000))),int(2602*(float(1067)/float(4000)))),(0,125,255),1)
cv2.imshow('Petri Dish ML Detections',img_dish_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
fromCenter=False
emb_missed=int(input('Missed single embryos = '))
for emb in range(emb_missed):
    (x_ss,y_ss,ws,hs)=cv2.selectROI('Petri Dish ML Detections',img_dish_new,fromCenter)
    cv2.rectangle(img_dish_new,(x_ss,y_ss),(x_ss+ws,y_ss+hs),(0,255,0),1)
    x_ss_1=int(x_ss*(float(6000)/float(1600)))
    x_ss_2=int((x_ss+ws)*(float(6000)/float(1600)))
    y_ss_1=int(y_ss*(float(4000)/float(1067)))
    y_ss_2=int((y_ss+hs)*(float(4000)/float(1067)))
    r,c=rowcol_fun(x_ss_1,x_ss_2,y_ss_1,y_ss_2)
    xc_rc.append([np.mean([x_ss_1,x_ss_2]),x_ss_1,x_ss_2,r,c])
    yc_rc.append([np.mean([y_ss_1,y_ss_2]),y_ss_1,y_ss_2,r,c])
emb_wrong=int(input('Wrong single embryos = '))
xc_rc_no=[]
yc_rc_no=[]
for emb in range(emb_wrong):
    (x_ss,y_ss,ws,hs)=cv2.selectROI('Petri Dish ML Detections',img_dish_new,fromCenter)
    cv2.rectangle(img_dish_new,(x_ss,y_ss),(x_ss+ws,y_ss+hs),(0,125,0),1)
    x_ss_1=int(x_ss*(float(6000)/float(1600)))
    x_ss_2=int((x_ss+ws)*(float(6000)/float(1600)))
    y_ss_1=int(y_ss*(float(4000)/float(1067)))
    y_ss_2=int((y_ss+hs)*(float(4000)/float(1067)))
    for emb_w in range(len(xc_rc)):
        if abs(int(np.mean([x_ss_1,x_ss_2]))-int(xc_rc[emb_w][0]))<10 and abs(int(np.mean([y_ss_1,y_ss_2]))-int(yc_rc[emb_w][0]))<10:
            xc_rc_no.append(xc_rc[emb_w])
            yc_rc_no.append(yc_rc[emb_w])
xc_rc_new = [e for e in xc_rc if e not in xc_rc_no]
yc_rc_new = [e for e in yc_rc if e not in yc_rc_no]

y1a_rc,y2a_rc,x1a_rc,x2a_rc=order(xc_rc_new,yc_rc_new,0,0)
img_dish=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
img_dish_new=cv2.resize(img_dish,(1600,1067))
for i in range(len(y1a_rc)):
    cv2.rectangle(img_dish,(x1a_rc[i][0],y1a_rc[i][0]),(x2a_rc[i][0],y2a_rc[i][0]),(0,255,0),5)
    cv2.rectangle(img_dish_new,(int(x1a_rc[i][0]*(float(1600)/float(6000))),int(y1a_rc[i][0]*(float(1067)/float(4000)))),(int(x2a_rc[i][0]*(float(1600)/float(6000))),int(y2a_rc[i][0]*(float(1067)/float(4000)))),(0,255,0),1)
cv2.imshow('Petri Dish ML Detections Final',img_dish_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/ML_Petri_Dishes/'+filename,img_dish)
print('Finished detecting embryos')
print('Number of embryos = {}'.format(len(y1a_rc)))

mypath='C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/Row_Col_Petri_Dish'
path='C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'
over=0
# Open new tensorflow session
graph = tf.Graph()
with graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile('C:/Users/User/Downloads/Andrew_files/faster_r_cnn_trained_model_injection_point_tip_pipette_robot_2_new_4'+'/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
with graph.as_default():
    with tf.compat.v1.Session() as sess:
        x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post,xc_rc_keep,yc_rc_keep=most_optimal_path(filename,y1a_rc,y2a_rc,x1a_rc,x2a_rc)
        positions=[]
        for i in range(len(xc_rc_keep)):
            embryo_point_center = function_transformation_matrix_DSLR_pipette(xc_rc_keep[i],yc_rc_keep[i],1963,1301,1349,2145,2554,2005,43680,27850,56780,9600,30780,12850) # DSLR to pipette
            if int(float(embryo_point_center.item(0,0)))<1000 or int(float(embryo_point_center.item(1,0)))<1000:
                print('Embryo out of reach')
            else:
                positions.append([int(float(embryo_point_center.item(0,0))),int(float(embryo_point_center.item(1,0)))])
        print(len(positions))
        for pic in range(len(positions)):
        # for pic in range(8,len(positions)):
            print('Embryo {} out of {} Embryos'.format(pic+1,len(positions)))
            # inj_depth=int(input('Injection depth = '))
            if pic==0:
            # if pic==8:
                print('New X = ',positions[pic][0])
                print('New Y = ',positions[pic][1])
                print('New Z = ',z_needle-300)
                dx_final=0
                dy_final=0
                dz=0
                view_1_x=431
                view_1_y=359
                view_2_x=633
                view_2_y=314
                time_wait=4.1
                print('Start air pressure')
                print('Start stream')
                footage_socket_1,footage_socket_2,z_needle_new,dx_final,dy_final,view_1_x,view_1_y,view_2_x,view_2_y=first_pipette(view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,footage_socket_1,footage_socket_2,inj_num,graph,sess,ser,Z_initial,pic,arduino)
                dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value_current,injection_time=injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle_new,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel)
            else:
                print('New X = ',positions[pic][0])
                print('New Y = ',positions[pic][1])
                print('New Z = ',Z_new )
                dist=math.hypot(positions[pic-1][0]-positions[pic][0],positions[pic-1][1]-positions[pic][1])
                print('Distance traveled = ',dist)
                if dist>10000:
                    Z_new=new_z(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,ser,pip_num,Z_inj_actual,pic)
                    dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value_current,injection_time=injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),Z_new,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel)
                else:
                    dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value_current,injection_time=injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),Z_new,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel)
            inj_num+=1
            pipette=0
            
            if injection_time!=0:
                injection_time_list.append(injection_time)
            if -10<int(sum_image)<sum_image_thresh_min:
                switch_list.append(1)
            elif int(sum_image)>=sum_image_thresh_min:
                switch_list.append(0)
            else:
                print('append nothing')
            if injected==2:
                missed+=1
                print('Missed injection')
                elim_embryo.append([current_x_centroid,current_y_centroid,current_z_centroid,pip_num,pic])
                print('Number of injected embryos = ',injected_embryos)
                print('Remove embryo from dish')
                print('Number of embryos missed = ',missed)
                injection_list.append(4) 
            elif int(sum_image)<sum_image_thresh_min:
                no_injected+=1
                print('No injection')
                elim_embryo.append([current_x_centroid,current_y_centroid,current_z_centroid,pip_num,pic])
                print('Number of injected embryos = ',injected_embryos)
                print('Remove embryo from dish')
                print('Number of embryos not injected = ',no_injected)
                injection_list.append(4) 
                injected_list.append(0) 
            else:
                if sum_image>sum_image_thresh_max and injected!=2 and int(sum_image)!=0:
                    inj_range=1
                    over_injected+=1
                    pressure_time=4
                    injected_embryos+=1
                    injected_embryos_count+=1
                    print('Number of injected embryos = ',injected_embryos)
                    print('Remove embryo from dish')
                    print('Number of embryos over injected = ',over_injected)
                    injection_list.append(4) 
                    injected_list.append(1) 
                elif sum_image_thresh_min<sum_image<sum_image_thresh_max and injected!=2 and int(sum_image)!=0:
                    inj_range=1
                    injection_list.append(injection_list_num)
                    print('Successful injection')
                    injected_embryos+=1
                    injected_embryos_count+=1
                    print('Number of injected embryos = ',injected_embryos)
                    injected_list.append(1)
                else:
                    print('Good')
            if len(switch_list)>2 and switch_list[len(switch_list)-3]==1 and switch_list[len(switch_list)-2]==1 and switch_list[len(switch_list)-1]==1:
            # if len(switch_list)>4 and switch_list[len(switch_list)-5]==1 and switch_list[len(switch_list)-4]==1 and switch_list[len(switch_list)-3]==1 and switch_list[len(switch_list)-2]==1 and switch_list[len(switch_list)-1]==1:
                print('CHANGE TO NEW PIPETTE AND VALVES!')
                inj_range=0
                pressure_value=1
                pressure_time=4
                time_wait=10
                injected_list=[]
                switch_list=[]
                switch=0
                pipette=1
                pip_num+=1
                set_0=0
                correct=0
                calib_pipette_num+=1
                pip_em_num.append(pic+1)
                dx_final,dy_final,current_x,current_y,current_z,footage_socket_1,footage_socket_2,Z_new,view_1_x,view_1_y,view_2_x,view_2_y,injected_embryos_count,dz_final,current_z_needle=new_pipette_new(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,ser,pip_num,Z_inj_actual,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,post_z,Z_initial,current_z_centroid,pic,sum_image_thresh_min,target_pixel) 
                deltas_pipette.insert(pip_num-1,[current_x-current_x_centroid,current_y-current_y_centroid,current_z_needle-current_z_centroid])
                if pic+1<len(positions):
                    XYZ_Location(20000,20000,8000,positions[pic+1][0]+int(dx_final),positions[pic+1][1]+int(dy_final),0,ser)
                    time.sleep(5)
                    XYZ_Location(20000,20000,8000,positions[pic+1][0]+int(dx_final),positions[pic+1][1]+int(dy_final),Z_new,ser)
                    time.sleep(5)
            print('Starting pressure = ',pressure_value)
        # Stop pressure
        time.sleep(1)
        arduino.write("P0p".encode())
        time.sleep(1)
        if injected_embryos==len(positions):
            XYZ_Location(20000,20000,8000,current_x,current_y,0,ser)
        else:
            if len(deltas_pipette)<2:
                print('no adding deltas')
            else:
                deltas_pipette_x=[]
                deltas_pipette_y=[]
                deltas_pipette_z=[]
                for w in range(len(deltas_pipette)):
                    deltas_pipette_x.append(deltas_pipette[w][0])
                    deltas_pipette_y.append(deltas_pipette[w][1])
                    deltas_pipette_z.append(deltas_pipette[w][2])
                for h in range(len(deltas_pipette)):
                    deltas_pipette[h]=[sum(deltas_pipette_x[h:len(deltas_pipette_x)]),sum(deltas_pipette_y[h:len(deltas_pipette_y)]),sum(deltas_pipette_z[h:len(deltas_pipette_z)])]
            elim_embryo_new=[]
            for q in range(len(elim_embryo)):
                elim_embryo_new.append([elim_embryo[q][0]+deltas_pipette[elim_embryo[q][3]][0],elim_embryo[q][1]+deltas_pipette[elim_embryo[q][3]][1],elim_embryo[q][2]+deltas_pipette[elim_embryo[q][3]][2],elim_embryo[q][3],elim_embryo[q][4]])
            injection_results_new(elim_embryo_new,filename,view_1_x,view_1_y,view_2_x,view_2_y,dx_final,dy_final,footage_socket_1,footage_socket_2,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,current_x_centroid,current_y_centroid,current_z_centroid,len(positions))
print('Press y on video stream')
# Close sockets
footage_socket_1.close()
footage_socket_2.close()
# Disconnect xyz stage
ser.close()
# Save petri dish and requisite injections          
detections_dslr_image(path,filename,mypath,injection_list,x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post)
total_end_time=time.time()
print('Number of injected embryos = ',injected_embryos)
print('{} % of dish injected'.format(float(injected_embryos)/float(len(positions))*100))
print('Average time for injection (s) = ',np.mean(injection_time_list))
print('Time for injection of dish (min) = ',int((total_end_time-total_start_time)/60)) 
print('Injection pressure (psi) = ',pressure_value)
print('Injection pressure time (s) = ',pressure_time)
print('Injection depth (um) = ',inj_depth)
print('Injection speed (um/s) = ',inj_speed)
