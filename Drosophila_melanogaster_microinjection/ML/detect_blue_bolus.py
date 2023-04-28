# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:48:43 2021

@author: User
"""

import cv2
from ml_injection_point_estimation_new_cody import ml_injection_point_estimation_new_cody
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from sort_files import sort_files
import numpy as np
import math 
# widths=[3200,3200,3200,1920,1920,1920,1920,1920,3200]
# heights=[2880,2880,2880,2400,2400,2400,2400,2400,2880]
min_=[1,2,3,4,7,8]
graph = tf.Graph()
with graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v2.io.gfile.GFile('C:/Users/me-alegr011-admin/Downloads/Robot_code/faster_r_cnn_trained_model_larvae'+'/frozen_inference_graph.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
with graph.as_default():
    with tf.compat.v1.Session() as sess:
        detection_coords_classes_final=[]
        for w in range(6):
            thresh_ml = .9
            img_list=[]
            res_list=[]
            mask_list=[]
            path='C:/Users/me-alegr011-admin/Pictures/Camera Roll/RSY'
            files = [ i for i in listdir(path) if isfile(join(path,i)) ]
            files_final=sort_files(files)
            img = cv2.imread(path+'/'+files_final[w][1], 1)
            img_list.append(img)
            output_dict_detection_boxes_stored, output_dict_detection_classes_stored, output_dict_detection_scores_stored, y1a_rc, y2a_rc, x1a_rc, x2a_rc, xc_rc, yc_rc,detection_coords_classes_all = ml_injection_point_estimation_new_cody([img], thresh_ml, 4000, 6000, graph, sess, 1)
            img_dish = cv2.imread('C:/Users/me-alegr011-admin/Pictures/Camera Roll/RSY/30_{}.jpg'.format(min_[w]), 1)
            img_dish_new = cv2.imread('C:/Users/me-alegr011-admin/Pictures/Camera Roll/RSY/30_{}.jpg'.format(min_[w]), 1)
            larvae_bolus=0
            larvae_non_bolus=0
            for i in range(len(y1a_rc)):
                for j in range(len(y1a_rc[i])):
                    if math.hypot(np.mean([y1a_rc[i][j],y2a_rc[i][j]])-1880,np.mean([x1a_rc[i][j],x2a_rc[i][j]])-2010)>1750:
                        print('Detection out of range')
                    else:
                        img_dish_crop=img_dish[int(y1a_rc[i][j]):int(y2a_rc[i][j]),int(x1a_rc[i][j]):int(x2a_rc[i][j])]
                        hsv = cv2.cvtColor(img_dish_crop, cv2.COLOR_BGR2HSV)
                        # lower_blue = np.array([40,40,40])
                        lower_blue = np.array([35,35,35])
                        upper_blue = np.array([130,255,255])
                        mask = cv2.inRange(hsv, lower_blue, upper_blue)
                        mask_list.append(mask)
                        sum_image=sum(sum(mask))
                        # cv2.imshow('frame',img_dish_crop)
                        # cv2.imshow('mask',mask)
                        # # cv2.imshow('res',res)
                        # cv2.waitKey(0)
                
                        if sum_image>0:
                            detection_coords_classes_all[j][4]=1
                            cv2.rectangle(img_dish_new,(int(x1a_rc[i][j]),int(y1a_rc[i][j])),(int(x2a_rc[i][j]),int(y2a_rc[i][j])),(255,255,0),3)
                            larvae_bolus+=1
                        else:
                            detection_coords_classes_all[j][4]=2
                            cv2.rectangle(img_dish_new,(int(x1a_rc[i][j]),int(y1a_rc[i][j])),(int(x2a_rc[i][j]),int(y2a_rc[i][j])),(0,255,0),3)
                            larvae_non_bolus+=1
            detection_coords_classes_final.append(detection_coords_classes_all)
            cv2.putText(img_dish_new, '# Larvae Non-Bolus = {}'.format(larvae_non_bolus), (4450,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,0), 2,cv2.LINE_AA)
            cv2.putText(img_dish_new, '# Larvae Bolus = {}'.format(larvae_bolus), (4450,400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 2,cv2.LINE_AA)
            cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/larvae_images_rsy/image_{}.jpg'.format(min_[w]),img_dish_new)

# np.save('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/detection_classes_high_res/detection_coords_classes_final.npy',detection_coords_classes_final)
# np.save('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/detection_classes_hard_food/detection_coords_classes_final.npy',detection_coords_classes_final)
# np.save('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/detection_classes_hard_food_bottom/detection_coords_classes_final.npy',detection_coords_classes_final)
