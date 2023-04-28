# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:19:12 2020

@author: enet-joshi317-admin
"""
import cv2
from CameraWorkbench_master.camera import AmscopeCamera
import time
def image_embryo_10x(e):
    S=AmscopeCamera(0)
    S.activate()
    time.sleep(2)
    frame_num=0
    while(True):
        frame=S.get_frame()
        cv2.circle(frame, (640,480), 5, (0, 0, 255) , -2)
        cv2.circle(frame, (562,729), 5, (0, 255, 0) , -2)

        cv2.circle(frame, (643,750), 5, (0, 0, 255) , -2)
        cv2.imshow('img1',frame) #display the captured image
        #cv2.imwrite('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/Embryo_4x_Dish/embryo_4x_{}.jpg'.format(e+1),frame)
        #frame=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/Embryo_4x_Dish/embryo_4x_{}.jpg'.format(e+1),1)
        frame_num+=1
        cv2.waitKey(1)
        
        if frame_num>10:
            time.sleep(1)
            cv2.destroyAllWindows()
            break
    S.close()
    return frame
image_embryo_10x(8)