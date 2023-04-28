# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:05:38 2022

@author: User
"""

import time
from XYZ_Stage.XYZ_Position import XYZ_Location
from injection_ml_tip_short_centroid_new import injection_ml_tip_short_centroid_new
import math
import serial

def injection_results_new(elim_embryo,filename,view_1_x,view_1_y,view_2_x,view_2_y,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,current_x_centroid,current_y_centroid,current_z_centroid,total_embryos):
    V=math.hypot(20000,20000)
    inv_V=(float(float(1)/float(V)))
    for i in range(len(elim_embryo)):
        # if i>0:
        #     # time_wait=(math.hypot(elim_embryo[i-1][0]-elim_embryo[i][0],elim_embryo[i-1][1]-elim_embryo[i][1])*inv_V)
        #     time_wait=2
        # else:
        #     time_wait=2
        #     # time_wait=5
        time_wait=2
        for k in range(2):
            dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,arduino,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid=injection_ml_tip_short_centroid_new(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,elim_embryo[i][0],elim_embryo[i][1],elim_embryo[i][2],dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,elim_embryo[i][4])    
            elim_embryo[i][0]=current_x_centroid
            elim_embryo[i][1]=current_y_centroid
            elim_embryo[i][2]=current_z_centroid
        XYZ_Location(20000,20000,8000,current_x_centroid,current_y_centroid,current_z_centroid+1000,ser)
        time.sleep(1)
        XYZ_Location(20000,20000,8000,current_x_centroid,current_y_centroid,current_z_centroid,ser)
        time.sleep(1)
        
        # XYZ_Location(20000,20000,8000,elim_embryo[i][0],elim_embryo[i][1],elim_embryo[i][2],ser)
        # time.sleep(2)
        # XYZ_Location(20000,20000,8000,elim_embryo[i][0],elim_embryo[i][1],elim_embryo[i][2]+1000,ser)
        # time.sleep(2)
        # XYZ_Location(20000,20000,8000,elim_embryo[i][0],elim_embryo[i][1],elim_embryo[i][2],ser)
        # time.sleep(1)
        
    # Bring back to 0
    # XYZ_Location(10000,10000,8000,current_x_centroid,current_y_centroid,0,ser)
    XYZ_Location(10000,10000,8000,elim_embryo[i][0],elim_embryo[i][1],0,ser)
    print(filename[7:22])
    print('Total embryos = ', total_embryos)

# # # 12 uninjected. 42 injected
# import zmq
# import numpy as np 
# import tensorflow as tf

# elim_embryo=[[30377.19225972281, 21789.550147900613, 21839, 0, 0],
#  [32048.89229674783, 21646.010561146788, 21995, 0, 2],
#  [33661.42029100248, 23560.194315520992, 21983, 0, 4],
#  [37954.47353470464, 24574.960480325517, 22053, 0, 8],
#  [43086.87480114603, 24046.284249892593, 22255, 0, 12],
#  [43229.465183498614, 17693.704366462684, 22244, 0, 25],
#  [39115.95355055551, 19760.14538549621, 22107, 0, 32],
#  [36761.93073299632, 16591.23362804532, 22146, 0, 36],
#  [36421.37867734555, 15060.480678113681, 22170, 0, 37],
#  [35803.28207203239, 14003.913041978205, 22059, 0, 39],
#  [34760.38920910692, 19605.713852465058, 22071, 0, 40],
#  [33607.059752067114, 20880.165180547676, 22028, 0, 41],
#  [33323.83664760119, 15893.142628195039, 21982, 1, 42],
#  [33427.02555869356, 17575.400941759734, 22065, 1, 43],
#  [31135.456974370034, 18890.510580617996, 21962, 1, 46],
#  [30433.571444275978, 15659.724109891755, 21994, 1, 48],
#  [29706.327134792176, 18357.022152392892, 21947, 1, 49],
#  [35309.173574622975, 12510.347379039587, 22072, 1, 50],
#  [41203.55890511623, 11722.039419992732, 22657, 2, 52]]
# ser = serial.Serial('COM3', 9600,timeout = 5)
# # Connect to arduino
# arduino = serial.Serial('COM7', 9600, timeout = 5)
# print('Connecting to arduino')
# context = zmq.Context()
# footage_socket_1 = context.socket(zmq.SUB)
# footage_socket_1.bind('tcp://*:5555')
# footage_socket_1.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
# footage_socket_2 = context.socket(zmq.SUB)
# footage_socket_2.bind('tcp://*:4555')
# footage_socket_2.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
# graph = tf.Graph()
# with graph.as_default():
#   od_graph_def = tf.compat.v1.GraphDef()
#   with tf.compat.v2.io.gfile.GFile('C:/Users/User/Downloads/Andrew_files/faster_r_cnn_trained_model_injection_point_tip_pipette_robot_2_new_4'+'/frozen_inference_graph.pb', 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
# with graph.as_default():
#     with tf.compat.v1.Session() as sess:
#         injection_results_new(elim_embryo,'filename',365,195,621,170,229,-448,footage_socket_1,footage_socket_2,55,graph,sess,arduino,0,0,0,0,2000,0,0,ser,0,0,46739,12279,22798,55)