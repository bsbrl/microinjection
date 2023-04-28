# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:32:34 2022

@author: User
"""

import cv2

def stream_function_multi_process(q,r):
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
    
    while(True):
        grabbed_1, frame_1 = cap_1.read()  # grab the current frame
        grabbed_2, frame_2 = cap_2.read()  # grab the current frame
        cv2.imshow("Stream 1", frame_1)
        cv2.imshow("Stream 2", frame_2)
        put=r.get()
        if put[0]==1:
            q.put([frame_1,frame_2])
        if put[0]==2:
            cap_1.release()
            cap_2.release()
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
            cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/injection_images/view_1.jpg',frame_1)
            cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/injection_images/view_2.jpg',frame_2)
            cap_1.release()
            cap_2.release()
            cv2.destroyAllWindows()
            break