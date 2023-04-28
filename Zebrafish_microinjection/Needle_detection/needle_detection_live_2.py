# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:11:20 2019

@author: enet-joshi317-admin
"""



import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

def needle_detection_live_2(img):
    # cap = cv2.VideoCapture(number)
    # if not(cap.isOpened()):
    #     print('Could not open video device')
    # else:
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*2)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
        # cap.set(cv2.CAP_PROP_FPS,30)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS,1)
        
        # ret, frame = cap.read()
        
        # img = cv2.imread('Needle_image.jpg')
        blur = cv2.GaussianBlur(img,(3,3),0)
        gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,0,30,3)
        # plt.figure(1)
        # plt.title('Canny Edge Detection')
        # plt.xlabel('x coordinate (pixels)')
        # plt.ylabel('y coordinate (pixels)')
        # plt.imshow(edges,cmap = 'gray')
        # plt.show()
        
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        x_edge=[]
        y_edge=[]
        for alll in range(len(contours)):
            for j_new in range(len(contours[alll])):
                x_edge.append((contours[alll][j_new][0][0]))
                y_edge.append((contours[alll][j_new][0][1]))
        # x_edge.remove(649)
        
        for g in range(1):
            # print(min(y_edge))
            if min(y_edge, default="EMPTY") in y_edge:
                y_edge.remove(min(y_edge))
        y_coord=min(y_edge, default=0)
        if min(y_edge, default=0) in y_edge:
            index_y_max=y_edge.index(y_coord)
        index_y_max = 0
        # print('y',y_coord)
        
        
        x_list=[]
        for r in range(3):
            if index_y_max+r >= len(x_edge):
                print('Needle 2 detection error')
            else:
                x_list.append(x_edge[index_y_max+r])
        # x_list=[]
        # for r in range(5):
        #     x_list.append(x_edge[index_y_max+r])
        
        if not x_list:
            x_ans = 0
        else:
            x_ans=int(np.mean(x_list))
        # x_ans=(max(x_edge) + min(x_edge))/2
        # print(max(x_edge))
        # print(min(x_edge))
        # print('x',x_ans)
        img = cv2.circle(img, (x_ans,y_coord), 2, (0, 0, 255), 2)
        # cv2.imwrite('Needle_image_detected.jpg',img)
        # cap.release()
        # time.sleep(2)
        # np.save('x_needle.npy',x_ans)
        # np.save('y_needle.npy',y_coord+60)
        
        # plt.figure(1)
        # plt.title('Original Image')
        # plt.xlabel('x coordinate (pixels)')
        # plt.ylabel('y coordinate (pixels)')
        # plt.imshow(img,cmap = 'gray')  
        # plt.figure(2)
        # plt.title('Canny Edge Detection')
        # plt.xlabel('x coordinate (pixels)')
        # plt.ylabel('y coordinate (pixels)')
        # plt.imshow(edges,cmap = 'gray')
        # plt.figure(3)
        # plt.title('Original Image With Tip Coordinate')
        # plt.xlabel('x coordinate (pixels)')
        # plt.ylabel('y coordinate (pixels)')
        # plt.plot(x_ans,y_coord,'ro',markersize=2)
        # plt.imshow(blur,cmap = 'gray')
        return (x_ans, y_coord, img)
    
# image_path = 'D:/Microinjection_Project/Python_Code/Needle_detection/image_captured.jpg'
# needle_detection(image_path)