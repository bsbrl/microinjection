# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:14:45 2020

@author: Andrew
"""


import cv2
#from new_function_ml_new_dslr import ml
#import numpy as np
def detections_dslr_image(path,filename,test_images_folder,injection_list,x1a_rc,y1a_rc,x2a_rc,y2a_rc):
    img_dish = cv2.imread(path+filename,1)
    for h in range(len(injection_list)):
        if injection_list[h]==0:
            cv2.rectangle(img_dish,(x1a_rc[h],y1a_rc[h]),(x2a_rc[h],y2a_rc[h]),(0,255,0),1)
            cv2.putText(img_dish,'# {}'.format(h+1),(int(x1a_rc[h]),int(y1a_rc[h])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        elif injection_list[h]==1:
            cv2.rectangle(img_dish,(x1a_rc[h],y1a_rc[h]),(x2a_rc[h],y2a_rc[h]),(0,0,255),1)
            cv2.putText(img_dish,'# {}'.format(h+1),(int(x1a_rc[h]),int(y1a_rc[h])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        elif injection_list[h]==2:
            cv2.rectangle(img_dish,(x1a_rc[h],y1a_rc[h]),(x2a_rc[h],y2a_rc[h]),(255,255,255),1)
            cv2.putText(img_dish,'# {}'.format(h+1),(int(x1a_rc[h]),int(y1a_rc[h])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        elif injection_list[h]==3:
            cv2.rectangle(img_dish,(x1a_rc[h],y1a_rc[h]),(x2a_rc[h],y2a_rc[h]),(255,255,0),1)
            cv2.putText(img_dish,'# {}'.format(h+1),(int(x1a_rc[h]),int(y1a_rc[h])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        elif injection_list[h]==4:
            cv2.rectangle(img_dish,(x1a_rc[h],y1a_rc[h]),(x2a_rc[h],y2a_rc[h]),(0,255,255),1)
            cv2.putText(img_dish,'# {}'.format(h+1),(int(x1a_rc[h]),int(y1a_rc[h])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        else:
            continue
    cv2.rectangle(img_dish,(400,180),(3340,3440),(0,125,255),1)
    cv2.putText(img_dish,'No Attempt',(4100,350),cv2.FONT_HERSHEY_SIMPLEX,5,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(img_dish,'Injected Embryo Posterior',(4100,550),cv2.FONT_HERSHEY_SIMPLEX,5,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img_dish,'Injected Embryo Anterior',(4100,750),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img_dish,'Injected Embryo Centroid',(4100,950),cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(img_dish,'Missed Injected Embryo',(4100,1150),cv2.FONT_HERSHEY_SIMPLEX,5,(0,255,255),2,cv2.LINE_AA)
    cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/DSLR_Camera/Post_Injection_Dish/'+filename,img_dish)
#path='C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'
#filename='Entire_Petri_Dish_304.jpg'
#test_images_folder='C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/Row_Col_Petri_Dish'
##injection_list=[1,3,1]
##number_of_pics=150
##thresh_ml=.95
##output_dict_detection_boxes_stored,output_dict_detection_classes_stored,output_dict_detection_scores_stored,embryo_count,y1a_rc,y2a_rc,x1a_rc,x2a_rc,xc_rc,yc_rc,rowcol_list,onlyfiles_new,single_embryos_count=ml('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/ML/faster_r_cnn_try_38','C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/ML/training_new_trial_embryo_nonembryo_clump_2','C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/Row_Col_Petri_Dish',number_of_pics,thresh_ml)
##x1a_rc=[[1275, 4, 4, 1], [1931, 5, 5, 1], [2058, 7, 6, 1]]
##y1a_rc=[[1459, 4, 4, 1], [1801, 5, 5, 1], [2684, 7, 6, 1]]
##x2a_rc=[[1303, 4, 4, 1], [1964, 5, 5, 1], [2084, 7, 6, 1]]
##y2a_rc=[[1497, 4, 4, 1], [1828, 5, 5, 1], [2730, 7, 6, 1]]
#detections_dslr_image(path,filename,test_images_folder,injection_list,x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post)