# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:23:43 2022

@author: me-alegr011-admin
"""

import serial
import time
from stream_image import stream_image
from Pressure_Control.Continuous_Pressure import continuous_pressure
import zmq
import numpy as np

arduino = serial.Serial('COM9', 9600, timeout = 5)
time.sleep(5)

context = zmq.Context()
footage_socket_1 = context.socket(zmq.SUB)
footage_socket_1.bind('tcp://*:5555')
footage_socket_1.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
footage_socket_2 = context.socket(zmq.SUB)
footage_socket_2.bind('tcp://*:4555')
footage_socket_2.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

for l in range(10):
    back_pressure_value_new=0
    pressure_value=1
    injected=2
    count=0
    while pressure_value<=40 and injected==2:
        count+=1
        correct=0
        o=0
        while correct==0 and injected==2:
            print('Try ',o+1)
            arduino.flush()
            signal=continuous_pressure(back_pressure_value_new,pressure_value,'inj')
            arduino.write(signal.encode())
            q_=arduino.readline()
            arduino.flush()
            q_=q_.decode()
            s=q_.find('Received')
            img1,img2=stream_image(footage_socket_1,footage_socket_2,1,0)
            arduino.flush()
                    
            if pressure_value>5:
                press_num=int((pressure_value + 43.6279)/0.9535)
            else:
                press_num=int((pressure_value + 0.35)/0.107)
            if pressure_value>5:
                if press_num>99:
                    if q_[s+9]=='P' and q_[s+10:s+13]==str(int((pressure_value + 43.6279)/0.9535)) and q_[s+13]=='p' and q_[s+14]=='\r':
                        correct=1
                    else:
                        o+=1
                else:
                    if q_[s+9]=='P' and q_[s+10:s+12]==str(int((pressure_value + 43.6279)/0.9535)) and q_[s+12]=='p' and q_[s+13]=='\r':
                        correct=1
                    else:
                        o+=1 
            else:
                if press_num>9:
                    if q_[s+9]=='P' and q_[s+10:s+12]==str(int((pressure_value + 0.35)/0.107)) and q_[s+12]=='p' and q_[s+13]=='\r':
                        correct=1
                    else:
                        o+=1 
                else:
                    if q_[s+9]=='P' and q_[s+10:s+11]==str(int((pressure_value + 0.35)/0.107)) and q_[s+11]=='p' and q_[s+12]=='\r':
                        correct=1
                    else:
                        o+=1 
        pressure_value+=1