# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:11:20 2019

@author: enet-joshi317-admin
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time 

def needle_detection(image_location_needle,number_of_detections):
    points_1x=[]
    points_1y=[]
    points_2x=[]
    points_2y=[]
    
    for i in range(number_of_detections):
        print('Detection {}'.format(i+1))
        # Comment out when taking video
        cap_1=cv2.VideoCapture(0)
        cap_2=cv2.VideoCapture(1)
        
        cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,960)
        cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,1280)
        cap_1.set(cv2.CAP_PROP_FPS,30)
        cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
        
        cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
        cap_2.set(cv2.CAP_PROP_FPS,30)
        cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)
        frame_num=0
        
        while(True):
            ret_1,frame_1 = cap_1.read()
            ret_2,frame_2 = cap_2.read()
            
            image_height=1280
            image_width=960
            center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
            scale=1
            M_1 = cv2.getRotationMatrix2D(center,450, scale)
            cosine = np.abs(M_1[0, 0])
            sine = np.abs(M_1[0, 1])
            nW = int((image_height * sine) + (image_height * cosine))
            nH = int((image_height * cosine) + (image_width * sine))
            M_1[0, 2] += (nW / 2) - int((float(image_width))/(2))
            M_1[1, 2] += (nH / 2) - int((float(image_height))/(2))
            new_1=cv2.warpAffine(frame_1, M_1, (image_height, image_width)) 
        
            image_height=960
            image_width=1280
            center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
            M_2 = cv2.getRotationMatrix2D(center,180, scale)
            cosine = np.abs(M_2[0, 0])
            sine = np.abs(M_2[0, 1])
            nW = int((image_height * sine) + (image_height * cosine))
            nH = int((image_height * cosine) + (image_width * sine))
            M_2[0, 2] += (nW / 2) - int((float(image_width))/(2))
            M_2[1, 2] += (nH / 2) - int((float(image_height))/(2))
            new_2=cv2.warpAffine(frame_2, M_2, (image_height, image_width)) 
            
            cv2.imshow('view_1',new_1) #display the captured image
            cv2.imshow('view_2',new_2) #display the captured image
            frame_num+=1
            cv2.waitKey(1)
            if i==0:
                frame_limit=250
            else:
                frame_limit=100
            
            if frame_num>frame_limit: #save on pressing 'y'
                time.sleep(1)
                cv2.imwrite(image_location_needle+'/view_1.jpg',new_1)
                cv2.imwrite(image_location_needle+'/view_2.jpg',new_2)
                cv2.destroyAllWindows()
                break
        cap_1.release()
        cap_2.release()
    
        for needle in range(0,2):
            print(needle)
            img = cv2.imread(image_location_needle+'/view_{}.jpg'.format(needle+1))
            img_orig = cv2.imread(image_location_needle+'/view_{}.jpg'.format(needle+1))
            if needle==1:
                crop_img_new = img[240:700,150:800]
            else:
                crop_img_new = img[0:500,350:700]
            cv2.imwrite(image_location_needle+'/needle_{}_crop.jpg'.format(needle+1),crop_img_new)
            img = cv2.imread(image_location_needle+'/needle_{}_crop.jpg'.format(needle+1))
            blur = cv2.GaussianBlur(img,(3,3),0)
            gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
            if needle==1:
                edges = cv2.Canny(gray,0,40,3)
            else:
                edges = cv2.Canny(gray,0,30,3)
            plt.figure(1)
            plt.title('Canny Edge Detection')
            plt.xlabel('x coordinate (pixels)')
            plt.ylabel('y coordinate (pixels)')
            plt.imshow(edges,cmap = 'gray')
            plt.show()
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            x_edge=[]
            y_edge=[]
            for alll in range(len(contours)):
                for j_new in range(len(contours[alll])):
                    x_edge.append((contours[alll][j_new][0][0]))
                    y_edge.append((contours[alll][j_new][0][1]))
            if needle==1:
                if 649 in x_edge:
                    x_edge.remove(649)
            if needle==1:
                if 0 in x_edge:
                    x_edge.remove(0)
            if needle==0:
                if 959 in y_edge:
                    y_edge.remove(959)
            if needle==0:
                if 0 in y_edge:
                    y_edge.remove(0)
    #        x_edge.sort()
    #        y_edge.sort()
            for g in range(1):
                print(max(y_edge))
                y_edge.remove(max(y_edge))
            y_coord=max(y_edge)
            index_y_max=y_edge.index(y_coord)
            print('y',y_coord)
            if needle==1:
                x_list=[]
                for r in range(5):
                    x_list.append(x_edge[index_y_max+r])
            else:
                x_list=[]
                for r in range(5):
                    x_list.append(x_edge[index_y_max+r])            
    
            x_ans=int(np.mean(x_list))
            print(max(x_edge))
            print(min(x_edge))
            print('x',x_ans)
            
            plt.figure(1)
            plt.title('Original Image')
            plt.xlabel('x coordinate (pixels)')
            plt.ylabel('y coordinate (pixels)')
            plt.imshow(img_orig,cmap = 'gray')  
            plt.figure(2)
            plt.title('Canny Edge Detection')
            plt.xlabel('x coordinate (pixels)')
            plt.ylabel('y coordinate (pixels)')
            plt.imshow(edges,cmap = 'gray')
            plt.figure(3)
            plt.title('Original Image With Tip Coordinate')
            plt.xlabel('x coordinate (pixels)')
            plt.ylabel('y coordinate (pixels)')
            plt.plot(x_ans,y_coord,'ro',markersize=2)
            plt.imshow(blur,cmap = 'gray')
            
            if needle==1:
                plt.figure(4)
                plt.title('View 2 Tip Detected')
                plt.xlabel('x coordinate (pixels)')
                plt.ylabel('y coordinate (pixels)')
                plt.plot(x_ans+150,y_coord+240,'ro',markersize=2)
                plt.imshow(img_orig,cmap = 'gray')
                plt.show()
                print('X tip coord, Y tip coord',[x_ans+150,y_coord+240])
                points_2x.append(x_ans+150)
                points_2y.append(y_coord+240)
            else:
                plt.figure(5)
                plt.title('View 1 Tip Detected')
                plt.xlabel('x coordinate (pixels)')
                plt.ylabel('y coordinate (pixels)')
                plt.plot(x_ans+350,y_coord,'ro',markersize=2)
                plt.imshow(img_orig,cmap = 'gray')
                plt.show()
                print('X tip coord, Y tip coord',[x_ans+350,y_coord])
                points_1x.append(x_ans+350)
                points_1y.append(y_coord)
    for j in range(number_of_detections):
        points_avg=[[np.mean(points_1x),np.mean(points_1y)],[np.mean(points_2x),np.mean(points_2y)]]
    return points_avg
start = time.time()
points=needle_detection('D:/Microinjection_Project/Python_Code/Needle_detection',5)
end = time.time()
print('Time to run = ',end-start)