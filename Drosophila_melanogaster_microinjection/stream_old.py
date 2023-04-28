# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:50:43 2021

@author: User
"""

import base64
import cv2
import zmq
# import numpy as np

# context = zmq.Context()
# footage_socket_1 = context.socket(zmq.PUB)
# footage_socket_1.connect('tcp://localhost:5555')
# footage_socket_2 = context.socket(zmq.PUB)
# footage_socket_2.connect('tcp://localhost:4555')
# Open Camera
cap_1=cv2.VideoCapture(0)
cap_2=cv2.VideoCapture(1)
cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap_1.set(cv2.CAP_PROP_FPS,30)
cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap_2.set(cv2.CAP_PROP_FPS,30)
cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)

inj_num=int(input('Dish number = '))
# inj_num=1
# inj_num=7
# video1 = cv2.VideoWriter('C:/Users/User/Downloads/Andrew_files/Injection_videos/injection_view_1_{}.avi'.format(inj_num), 0, 15, (1280,720))
# video2 = cv2.VideoWriter('C:/Users/User/Downloads/Andrew_files/Injection_videos/injection_view_2_{}.avi'.format(inj_num), 0, 15, (1280,720))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video1 = cv2.VideoWriter('D:/injection_view_1_dish_{}.mp4'.format(inj_num), fourcc, 30, (1280,720))
video2 = cv2.VideoWriter('D:/injection_view_2_dish_{}.mp4'.format(inj_num), fourcc, 30, (1280,720))
video_on=1
# context = zmq.Context()
# footage_socket_1 = context.socket(zmq.PUB)
# footage_socket_1.connect('tcp://localhost:5555')
# footage_socket_2 = context.socket(zmq.PUB)
# footage_socket_2.connect('tcp://localhost:4555')
while video_on==1:
    context = zmq.Context()
    footage_socket_1 = context.socket(zmq.PUB)
    footage_socket_1.connect('tcp://localhost:5555')
    footage_socket_2 = context.socket(zmq.PUB)
    footage_socket_2.connect('tcp://localhost:4555')
    grabbed_1, frame_1 = cap_1.read()  # grab the current frame
    grabbed_2, frame_2 = cap_2.read()  # grab the current frame
    # frame_1_rs = cv2.resize(frame_1, (1280, 720))  # resize the frame
    # frame_2_rs = cv2.resize(frame_2, (1280, 720))  # resize the frame
    encoded_1, buffer_1 = cv2.imencode('.jpg', frame_1)
    encoded_2, buffer_2 = cv2.imencode('.jpg', frame_2)
    jpg_as_text_1 = base64.b64encode(buffer_1)
    jpg_as_text_2 = base64.b64encode(buffer_2)
    footage_socket_1.send(jpg_as_text_1)
    footage_socket_2.send(jpg_as_text_2)
    video1.write(frame_1)
    video2.write(frame_2)
    cv2.rectangle(frame_1,(400,300),(880,420),(0,0,0),1)
    cv2.rectangle(frame_2,(400,300),(880,420),(0,0,0),1)
    cv2.imshow("Stream 1", frame_1)
    cv2.imshow("Stream 2", frame_2)
    # cv2.waitKey(1)
    footage_socket_1.close()
    footage_socket_2.close()
    if cv2.waitKey(1) & 0xFF == ord('y'):
        # footage_socket_1.close()
        # footage_socket_2.close()
        cap_1.release()
        cap_2.release()
        video1.release()
        video2.release()
        cv2.destroyAllWindows()
        video_on=0
        # frame_1 = np.zeros([1280,720])
        # encoded, buffer = cv2.imencode('.jpg', frame_1)
        # jpg_as_text = base64.b64encode(buffer)
        # footage_socket_1.send(jpg_as_text)
        # break