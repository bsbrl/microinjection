# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:01:40 2021

@author: User
"""

import cv2
import base64
import numpy as np
def stream_image(footage_socket_1,footage_socket_2):
    frame_1=footage_socket_1.recv_string()
    frame_2=footage_socket_2.recv_string()
    img_1 = base64.b64decode(frame_1)
    img_2 = base64.b64decode(frame_2)
    npimg_1 = np.fromstring(img_1, dtype=np.uint8)
    npimg_2 = np.fromstring(img_2, dtype=np.uint8)
    source_1 = cv2.imdecode(npimg_1,cv2.IMREAD_COLOR)
    source_2 = cv2.imdecode(npimg_2,cv2.IMREAD_COLOR)
    return source_1,source_2
# import zmq
# import time
# # Open sockets
# context = zmq.Context()
# footage_socket_1 = context.socket(zmq.PUB)
# footage_socket_1.connect('tcp://localhost:5555')
# footage_socket_2 = context.socket(zmq.PUB)
# footage_socket_2.connect('tcp://localhost:4555')
# f,m=stream_image(footage_socket_1,footage_socket_2)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/stream_test_images/image1_1.jpg',f)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/stream_test_images/image2_1.jpg',m)
# print('move')
# time.sleep(5)
# f,m=stream_image(footage_socket_1,footage_socket_2)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/stream_test_images/image1_2.jpg',f)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/stream_test_images/image2_2.jpg',m)
# print('move')
# time.sleep(5)
# f,m=stream_image(footage_socket_1,footage_socket_2)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/stream_test_images/image1_3.jpg',f)
# cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/stream_test_images/image2_3.jpg',m)
# time.sleep(5)
