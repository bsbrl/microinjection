# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:11:06 2020

@author: admin
"""

import time
import numpy as np
import cv2
import serial 
import os
import matplotlib.pyplot as plt
os.chdir('D:/Microinjection_Project/Python_Code/')
from CameraWorkbench_master.camera import *
from XYZ_stage.MyXYZ import MyXYZ

def autofocus(x, y, z_ref):
    S = AmscopeCamera(0)
    S.activate()
    for p in range(50):
        frame = S.get_frame()
    # time.sleep(3)
    XYZ = MyXYZ()
    total_z = 1
    if z_ref > (total_z/2):
        print('Autofocusing starts')
        step_dis = [0.25, 0.1, 0.05, 0.01]
        check_dis = [total_z, 6*step_dis[1], 8*step_dis[2], 8*step_dis[3]]
        for i in range(len(step_dis)):
            all_variance = []
            all_z = []
            for j in range(int(check_dis[i]/step_dis[i])+1):
                current_z = (z_ref - (check_dis[i]/2) + (j*step_dis[i]))
                XYZ.Position(x, y, current_z)
                frame = S.get_frame()
                frame = cv2.blur(frame, (5,5))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # cv2.imwrite('image_{}_{}.jpg'.format(i,j),frame)
                variance = cv2.Laplacian(frame, cv2.CV_64F).var()
                print(variance)
                all_z.append(current_z*1000)
                all_variance.append(variance)
            time.sleep(0.5)
            # print(all_z)
            # print(all_variance)
            index = np.argmax(all_variance)
            z_ref = all_z[index]/1000
            print(z_ref)
            plt.clf()
            fig = plt.figure(i+1)
            plt.plot(all_z, all_variance)
            fig.suptitle('x = {:.2f} um; y = {:.2f} um; z = {:.2f} um'.format(x, y, z_ref))
            plt.xlabel('Z axis (um)')
            plt.ylabel('Variance')
            plt.savefig('Variance_{}.png'.format(i+1), dpi=300)
            plt.clf()
        XYZ.Position(x, y, z_ref)
        frame = S.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('focused_image.jpg',frame) 
        S.deactivate()
    else:
        print('Reference z value should be greater than', total_z/2)
    return z_ref
        

# autofocus(12.3,9.2,13)