# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:19:12 2020

@author: enet-joshi317-admin
"""
import cv2
from CameraWorkbench_master.camera import *
import time

def image_embryo_10x():
    S=AmscopeCamera(0)
    S.activate()
    time.sleep(2)
    # S.set_exposure(5)
    # S.set_gain(100)
    # S.set_brightness(0)
    # S.set_contrast(0)
    while(True):
        frame=S.get_frame()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.circle(frame, (640,480), 5, (0, 0, 255) , -2)
        cv2.imshow('img1',frame) #display the captured image
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
            time.sleep(1)
            cv2.imwrite('embryo.jpg',frame)
            cv2.destroyAllWindows()
            S.deactivate()
            break
        
image_embryo_10x()