# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:19:15 2021

@author: User
"""

import numpy as np
import sys
import tensorflow as tf
import time
# from matplotlib import pyplot as plt
# import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.10':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

def ml_injection_point_estimation_new(image,ml_threshold,im_height,im_width,detection_graph,sess,step):   
    def load_image_into_numpy_array(image,im_height,im_width):
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)
      
    # # Detection
    # In[50];
    def run_inference_for_multiple_images(images, graph,sess):
      with graph.as_default():
          output_dicts = []     
          for index, image in enumerate(images):
              ops = tf.compat.v1.get_default_graph().get_operations()
              all_tensor_names = {output.name for op in ops for output in op.outputs}
              tensor_dict = {}
              for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                      tensor_name)
              if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, .5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
              image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
        
    #           Run inference
              output_dict = sess.run(tensor_dict,
                                     feed_dict={image_tensor: np.expand_dims(image, 0)})        
              # all outputs are float32 numpy arrays, so convert types as appropriate
              output_dict['num_detections'] = int(output_dict['num_detections'][0])
              output_dict['detection_classes'] = output_dict[
                  'detection_classes'][0].astype(np.uint8)
              output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
              output_dict['detection_scores'] = output_dict['detection_scores'][0]
              if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    
              output_dicts.append(output_dict)
    
      return output_dicts
    start = time.time()
    output_dict_detection_scores_stored=[]
    output_dict_detection_boxes_stored=[]
    output_dict_detection_classes_stored=[]
    output_dicts = run_inference_for_multiple_images(image, detection_graph,sess)
        
#    IMAGE_SIZE = (5, 5)
    y1a_rc_list=[]
    y2a_rc_list=[]
    x1a_rc_list=[]
    x2a_rc_list=[]
    xc_rc_list=[]
    yc_rc_list=[]

    for r in range(len(output_dicts)):
        output_dict_detection_scores_stored.append(sorted([i for i in output_dicts[r]['detection_scores'] if i >= ml_threshold],reverse=True))
        output_dict_detection_boxes_stored.append(output_dicts[r]['detection_boxes'][0:len(output_dict_detection_scores_stored[r])])
        output_dict_detection_classes_stored.append(output_dicts[r]['detection_classes'][0:len(output_dict_detection_scores_stored[r])])

    for f in range(len(output_dict_detection_boxes_stored)):
        if len(output_dict_detection_boxes_stored[f])!=0:
            y1a_rc=[]
            y2a_rc=[]
            x1a_rc=[]
            x2a_rc=[]
            xc_rc=[]
            yc_rc=[]
            for i in range(len(output_dict_detection_boxes_stored[f])):
                y1=int(output_dict_detection_boxes_stored[f][i][0]*im_height)
                y2=int(output_dict_detection_boxes_stored[f][i][2]*im_height)
                x1=int(output_dict_detection_boxes_stored[f][i][1]*im_width)
                x2=int(output_dict_detection_boxes_stored[f][i][3]*im_width)
                y1a_rc.append(y1)
                y2a_rc.append(y2)
                x1a_rc.append(x1)
                x2a_rc.append(x2)
                xc_rc.append(float(float(int(x2+x1))/float(2)))
                yc_rc.append(float(float(int(y2+y1))/float(2)))
            y1a_rc_list.append(y1a_rc)
            y2a_rc_list.append(y2a_rc)
            x1a_rc_list.append(x1a_rc)
            x2a_rc_list.append(x2a_rc)
            yc_rc_list.append(yc_rc)
            xc_rc_list.append(xc_rc)
        else:
            print('No injection point detect')
    end = time.time()
    print('Time to run (s) = ',end-start)
    print('second/image = ', float(end-start)/float(1))

    return output_dict_detection_boxes_stored, output_dict_detection_classes_stored, output_dict_detection_scores_stored, y1a_rc_list, y2a_rc_list, x1a_rc_list, x2a_rc_list, xc_rc_list, yc_rc_list

# from matplotlib import pyplot as plt
# import cv2
# from os import listdir
# from os.path import isfile, join

# # thresh_ml = .01
# # correct=[]
# path='C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Tip_test'
# files = [ i for i in listdir(path) if isfile(join(path,i)) ]
# graph = tf.Graph()
# with graph.as_default():
#   od_graph_def = tf.compat.v1.GraphDef()
#   with tf.compat.v2.io.gfile.GFile('C:/Users/me-alegr011-admin/Downloads/Robot_code/faster_r_cnn_trained_model_injection_point_tip_pipette_robot_2_new_4'+'/frozen_inference_graph.pb', 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
# with graph.as_default():
#     with tf.compat.v1.Session() as sess:
#         for i in range(2):
#             print('Image {}'.format(i+1))
#             img1 = cv2.imread(path+'/'+files[i], 1)
#             h, w, c = img1.shape
#             img1_list=[img1]
#             output_dict_detection_boxes_stored, output_dict_detection_classes_stored, output_dict_detection_scores_stored, y1a_rc, y2a_rc, x1a_rc, x2a_rc, xc_rc, yc_rc = ml_injection_point_estimation_new(img1_list, .01, h, w, graph, sess, 1)
#             list_classes=output_dict_detection_classes_stored[0].tolist()
#             if 5 in list_classes:
#                 list_classes_index=list_classes.index(5)
#                 crop=img1[y1a_rc[0][list_classes_index]:y2a_rc[0][list_classes_index],x1a_rc[0][list_classes_index]:x2a_rc[0][list_classes_index]]
#                 cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Tip_test/crop_{}.jpg'.format(i+1),crop)
#                 cv2.circle(img1, (int(xc_rc[0][list_classes_index]),int(yc_rc[0][list_classes_index])), 5, (0, 0, 255) , 2)
#                 cv2.imshow('view',img1) #display the captured image
#                 cv2.waitKey(0)
#                 print('ML tip x = ',xc_rc[0][list_classes_index])
#                 print('ML tip y = ',yc_rc[0][list_classes_index])

#                 lower_blue = np.array([52,30,35])
#                 upper_blue = np.array([255,255,255])
#                 hsv_1 = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#                 mask_1 = cv2.inRange(hsv_1, lower_blue, upper_blue)
#                 mask_1_list=mask_1.tolist()
#                 cv2.imshow('image',mask_1)
#                 cv2.waitKey(0)
#                 x_list=[]
#                 y_list=[]
#                 for j in range(len(mask_1)):
#                     indices = [i for i, x in enumerate(mask_1_list[j]) if x==255]
#                     if indices!=[]:
#                         x_list.append(np.median(indices))
#                         y_list.append(j)
#                 cv2.circle(img1, (int(x_list[len(x_list)-1]+x1a_rc[0][list_classes_index]),int(y_list[len(y_list)-1]+y1a_rc[0][list_classes_index])), 5, (0, 255, 255) , 2)
#                 cv2.imshow('view',img1) #display the captured image
#                 cv2.waitKey(0)
#                 print('CV tip x = ',x_list[len(x_list)-1]+x1a_rc[0][list_classes_index])
#                 print('CV tip y = ',y_list[len(y_list)-1]+y1a_rc[0][list_classes_index])
# # for i in range(2):
# #     print('Image {}'.format(i+1))
# #     img1 = cv2.imread(path+'/'+files[i], 1)
# #     h, w, c = img1.shape
# #     img1_list=[img1]
    
# #     crop_new=cv2.imread('C:/Users/me-alegr011-admin/Downloads/Robot_code/ML/Tip_test/crop_{}.jpg'.format(i+1),1)
# #     lower_blue = np.array([52,30,35])
# #     upper_blue = np.array([255,255,255])
# #     hsv_1 = cv2.cvtColor(crop_new, cv2.COLOR_BGR2HSV)
# #     mask_1 = cv2.inRange(hsv_1, lower_blue, upper_blue)
# #     mask_1_list=mask_1.tolist()
# #     cv2.imshow('image',mask_1)
# #     cv2.waitKey(0)
 
