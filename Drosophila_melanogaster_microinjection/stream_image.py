# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:01:40 2021

@author: User
"""

import cv2
import base64
import numpy as np
import zmq
import time

# def stream_image(footage_socket_1,footage_socket_2,inj_num,step_num):
#     context = zmq.Context()
#     footage_socket_3 = context.socket(zmq.PUB)
#     footage_socket_3.connect('tcp://localhost:3555')
#     # time.sleep(.25)
#     time.sleep(.1)
#     footage_socket_3.send(str(inj_num+1).encode('utf-8'))
#     check_1=0
#     check_2=0
#     while check_1==0 or check_2==0:
#         check_1=footage_socket_1.poll(1)
#         check_2=footage_socket_2.poll(1)
#     footage_socket_3.send(str('-1').encode('utf-8'))
#     footage_socket_3.close()
#     frame_1=footage_socket_1.recv_string()
#     frame_2=footage_socket_2.recv_string()
#     img_1 = base64.b64decode(frame_1)
#     img_2 = base64.b64decode(frame_2)
#     npimg_1 = np.fromstring(img_1, dtype=np.uint8)
#     npimg_2 = np.fromstring(img_2, dtype=np.uint8)
#     source_1 = cv2.imdecode(npimg_1,cv2.IMREAD_COLOR)
#     source_2 = cv2.imdecode(npimg_2,cv2.IMREAD_COLOR)

    
#     return source_1,source_2

def stream_image(footage_socket_1,footage_socket_2,footage_socket_3,inj_num,step_num):
    footage_socket_3.send(str(inj_num+1).encode('utf-8'))
    check_1=0
    check_2=0
    while check_1==0 or check_2==0:
        check_1=footage_socket_1.poll(1)
        check_2=footage_socket_2.poll(1)
    footage_socket_3.send(str('-1').encode('utf-8'))
    # footage_socket_3.close()
    frame_1=footage_socket_1.recv_string()
    frame_2=footage_socket_2.recv_string()
    img_1 = base64.b64decode(frame_1)
    img_2 = base64.b64decode(frame_2)
    npimg_1 = np.frombuffer(img_1, dtype=np.uint8)
    npimg_2 = np.frombuffer(img_2, dtype=np.uint8)
    source_1 = cv2.imdecode(npimg_1,cv2.IMREAD_COLOR)
    source_2 = cv2.imdecode(npimg_2,cv2.IMREAD_COLOR)

    
    return source_1,source_2

    
# import time
# # Open sockets
# context = zmq.Context()
# footage_socket_1 = context.socket(zmq.SUB)
# footage_socket_1.bind('tcp://*:5555')
# footage_socket_1.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
# footage_socket_2 = context.socket(zmq.SUB)
# footage_socket_2.bind('tcp://*:4555')
# footage_socket_2.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
# footage_socket_3 = context.socket(zmq.PUB)
# footage_socket_3.connect('tcp://localhost:3555')
# time.sleep(.1)
# print('on')
# time.sleep(5)
# f,m=stream_image(footage_socket_1,footage_socket_2,footage_socket_3,0,1)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/Process_images/1img1.jpg',f)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/Process_images/2img1.jpg',m)
# print('off')
# time.sleep(5)
# f,m=stream_image(footage_socket_1,footage_socket_2,footage_socket_3,2,1)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/Process_images/1img2.jpg',f)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/Process_images/2img2.jpg',m)
# print('on')
# time.sleep(5)
# f,m=stream_image(footage_socket_1,footage_socket_2,footage_socket_3,3,1)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/Process_images/1img3.jpg',f)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/Process_images/2img3.jpg',m)
