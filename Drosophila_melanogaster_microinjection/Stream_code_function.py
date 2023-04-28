# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:38:23 2022

@author: me-alegr011-admin

"""
import base64
import cv2
import zmq
import numpy as np
import serial
from XYZ_Stage.XYZ_Position import XYZ_Location

def stream_function_multi_process(q):
    try:
      ser = serial.Serial(
        port="COM5",
        baudrate=9600,
      )
      ser.isOpen() # try to open port, if possible print message and proceed with 'while True:'
      print ("port is opened!")
    
    except IOError: # if port is already opened, close it and open it again and print message
      ser.close()
      ser.open()
      print ("port was already open, was closed and opened again!")
    XYZ_Location(11250,11250,8000,43430,45000,5000,ser)
    
    context = zmq.Context()
    footage_socket_1 = context.socket(zmq.PUB)
    footage_socket_1.setsockopt(zmq.LINGER, 0)
    footage_socket_1.connect('tcp://localhost:5555')
    footage_socket_2 = context.socket(zmq.PUB)
    footage_socket_2.setsockopt(zmq.LINGER, 0)
    footage_socket_2.connect('tcp://localhost:4555')
    footage_socket_3 = context.socket(zmq.SUB)
    try:
        footage_socket_3.bind('tcp://*:3555')
    except:
        print('socket already in use')
    footage_socket_3.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    cap_1=cv2.VideoCapture(1)
    cap_2=cv2.VideoCapture(0)
    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_1.set(cv2.CAP_PROP_FPS,30)
    cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_2.set(cv2.CAP_PROP_FPS,30)
    cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)
    
    video_on=1
    inj_num=1
    while video_on==1:
        grabbed_1, frame_1 = cap_1.read()  # grab the current frame
        grabbed_2, frame_2 = cap_2.read()  # grab the current frame
        check=footage_socket_3.poll(1)
        if check==1:
            inj_num=footage_socket_3.recv()
            inj_num=int(inj_num.decode('utf-8'))
            encoded_1, buffer_1 = cv2.imencode('.jpg', frame_1)
            encoded_2, buffer_2 = cv2.imencode('.jpg', frame_2)
            jpg_as_text_1 = base64.b64encode(buffer_1)
            jpg_as_text_2 = base64.b64encode(buffer_2)
            rec=0
            while rec!=-1:
                footage_socket_1.send(jpg_as_text_1)
                footage_socket_2.send(jpg_as_text_2)
                check=footage_socket_3.poll(1)
                if check==1:
                    num=footage_socket_3.recv()
                    rec=int(num.decode('utf-8'))
        else:
            inj_num=inj_num
        inj_num=str(inj_num)
        cv2.putText(frame_1, 'Embryo '+inj_num, (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,cv2.LINE_AA)
        cv2.putText(frame_2, 'Embryo '+inj_num, (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,cv2.LINE_AA)
        frame_1 = cv2.resize(frame_1, (889, 500))  # resize the frame
        frame_2 = cv2.resize(frame_2, (889, 500))  # resize the frame
        if q.empty()==True:
            q.put([0,frame_1,frame_2])
    cap_1.release()
    cap_2.release()
    footage_socket_1.close()
    footage_socket_2.close()
    footage_socket_3.close()