# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:31:18 2021

@author: User
"""

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

def ml_injection_point_estimation_new_scores(image,ml_threshold,im_height,im_width,detection_graph,sess,step):   
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
    scores_list=[]
    
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
            scores=[]
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
                scores.append(output_dict_detection_scores_stored[f][i])
            y1a_rc_list.append(y1a_rc)
            y2a_rc_list.append(y2a_rc)
            x1a_rc_list.append(x1a_rc)
            x2a_rc_list.append(x2a_rc)
            yc_rc_list.append(yc_rc)
            xc_rc_list.append(xc_rc)
            scores_list.append(scores)
        else:
            print('No injection point detect')
    end = time.time()
    print('Time to run (s) = ',end-start)
    print('second/image = ', float(end-start)/float(1))

    return output_dict_detection_boxes_stored, output_dict_detection_classes_stored, output_dict_detection_scores_stored, y1a_rc_list, y2a_rc_list, x1a_rc_list, x2a_rc_list, xc_rc_list, yc_rc_list,scores_list

# from matplotlib import pyplot as plt
# import cv2
# from math import cos,pi,sin
# thresh_ml = .5
# #view_1_x=553
# #view_1_y=289
# #im_width=x2_crop-x1_crop
# #im_height=y2_crop-y1_crop

# view_1_x = 561
# view_1_y = 262
# view_2_x = 409
# view_2_y = 349
# x1_1_crop = int(view_1_x-300)
# x2_1_crop = int(view_1_x+300)
# y1_1_crop = int(view_1_y-300)
# y2_1_crop = int(view_1_y+300)


# x1_2_crop = int(view_2_x-300)
# x2_2_crop = int(view_2_x+300)
# y1_2_crop = int(view_2_y-300)
# y2_2_crop = int(view_2_y+300)

# if x1_1_crop < 0:
#     x1_1_crop = 0
# if y1_1_crop < 0:
#     y1_1_crop = 0
# if x2_1_crop > 1280:
#     x2_1_crop = 1280
# if y2_1_crop > 720:
#     y2_1_crop = 720

# if x1_2_crop < 0:
#     x1_2_crop = 0
# if y1_2_crop < 0:
#     y1_2_crop = 0
# if x2_2_crop > 1280:
#     x2_2_crop = 1280
# if y2_2_crop > 720:
#     y2_2_crop = 720

# # im_width_1 = x2_1_crop-x1_1_crop
# # im_height_1 = y2_1_crop-y1_1_crop
# # im_width_2 = x2_2_crop-x1_2_crop
# # im_height_2 = y2_2_crop-y1_2_crop
# img1 = cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/Post_injection_new/post_fov_1_1.jpg', 1)
# # image_height=600
# # image_width=800
# # center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
# # scale=1
# # fromCenter=False
# # M_1 = cv2.getRotationMatrix2D(center,90, scale)
# # cosine = np.abs(M_1[0, 0])
# # sine = np.abs(M_1[0, 1])
# # nW = int((image_height * sine) + (image_width * cosine))
# # nH = int((image_height * cosine) + (image_width * sine))
# # M_1[0, 2] += (nW / 2) - int((float(image_width))/(2))
# # M_1[1, 2] += (nH / 2) - int((float(image_height))/(2))
# # img1=cv2.warpAffine(img1, M_1, (image_height, image_width)) 
# # cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/img_1_2.jpg',img1)
# # BLUE = [255,255,255]
# # constant= cv2.copyMakeBorder(img1,200,0,155,0,cv2.BORDER_CONSTANT,value=BLUE)
# # cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/img_2_2.jpg', constant)
# # img1 = cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/img_2_2.jpg', 1)
# # img1=cv2.resize(img1,(800,600))
# # cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_divided_bolus_paint/temp_image_77_6_2.jpg',img1)
# # img1 = img1[200:568, 0:800]
# img1_list=[img1]
# #view_2_x=576
# #view_2_y=740
# #x1_2_crop=int(view_2_x-175)
# #x2_2_crop=int(view_2_x+175)
# #y1_2_crop=int(view_2_y-50)
# #y2_2_crop=int(view_2_y+300)
# # view_1_x=398
# # view_1_y=290
# graph = tf.Graph()
# with graph.as_default():
#   od_graph_def = tf.compat.v1.GraphDef()
#   #with tf.compat.v2.io.gfile.GFile('C:/Users/User/Downloads/Andrew_files/faster_r_cnn_trained_model_bolus_paint'+'/frozen_inference_graph.pb', 'rb') as fid:
#   with tf.compat.v2.io.gfile.GFile('C:/Users/User/Downloads/Andrew_files/faster_r_cnn_trained_model_injection_point_tip_pipette_robot_2_new_4'+'/frozen_inference_graph.pb', 'rb') as fid:
#     serialized_graph = fid.read()
#     od_graph_def.ParseFromString(serialized_graph)
#     tf.import_graph_def(od_graph_def, name='')
# with graph.as_default():
#     with tf.compat.v1.Session() as sess:
#         for i in range(1):
#           #            img1=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/ML/All_needle_images/view_2.jpg',1)
#           #            cv2.imwrite('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/ML/All_needle_images/view_2_crop.jpg',img1)
#             # img1=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/Post_injection/post_fov_1_2621.jpg',1)
#             # img1=img1[0:553,200:600]
#             # img1=convert_pipette_new(img1,view_1_x-50,0,view_1_x+50,view_1_y+10)
#             # img1=insert_blur_spot(img1,380,0,420,135,33)
#             output_dict_detection_boxes_stored, output_dict_detection_classes_stored, output_dict_detection_scores_stored, y1a_rc, y2a_rc, x1a_rc, x2a_rc, xc_rc, yc_rc,scores_list = ml_injection_point_estimation_new_scores(img1_list, thresh_ml, 600, 800, graph, sess, 1)

# # img_dish_new=cv2.resize(img1,(473,1000))
# # for i in range(len(y1a_rc)):
# #     cv2.rectangle(img_dish_new,(int(x1a_rc[i][0]*(float(473)/float(636))),int(y1a_rc[i][0]*(float(1000)/float(1344)))),(int(x2a_rc[i][0]*(float(473)/float(636))),int(y2a_rc[i][0]*(float(1000)/float(1344)))),(0,255,0),1)
# # cv2.imshow('Petri Dish ML Detections Final',img_dish_new)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # img1=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/inj_point_calib/view_1_p3.jpg',1)
# # img1=convert_pipette_new(img1,view_1_x-50,0,view_1_x+50,view_1_y+10)
# # img1=insert_blur_spot(img1,380,0,420,135,33)
# # cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/Blur/image.jpg',img1)
# # list_classes_1=output_dict_detection_classes_stored[0].tolist()
# # list_classes_index_1=list_classes_1.index(3)
# # x_new=(xc_rc[0][list_classes_index_1])*(image_height/800)
# # y_new=(yc_rc[0][list_classes_index_1])*(image_width/600)
# # xm=image_height/2
# # ym=image_width/2
# # a=-90*(pi/180)
# # x_post_1=(y_new-ym)*sin(a)+(x_new-xm)*cos(a)+ym
# # y_post_1=(y_new-ym)*cos(a)-(x_new-xm)*sin(a)+xm
# # img1 = cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/Injection_images/img_1_2.jpg', 1)
# plt.figure(2)
# plt.title('Detected Points')
# plt.xlabel('x coordinate (px)')
# plt.ylabel('y coordinate (px)')
# # plt.plot(xc_rc[0][4], yc_rc[0][4], 'ro', markersize=4)
# # plt.plot(x_post_1, y_post_1, 'ro', markersize=4)
# plt.plot(x1a_rc[0][0],y1a_rc[0][0],'bo',markersize=4)
# plt.plot(x2a_rc[0][0],y2a_rc[0][0],'co',markersize=4)
# # plt.plot(xc_rc[3],yc_rc[3],'yo',markersize=4)
# # plt.plot(xc_rc[4],yc_rc[4],'co',markersize=2)
# plt.imshow(img1, cmap='gray')
# plt.show()
# #plt.savefig('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/ML/All_needle_images/Figure_Images/ML_final_inj_tip_1.jpg',format='jpg')
# #plt.savefig('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/ML/All_needle_images/Figure_Images/ML_final_inj_tip_1.eps',format='eps')
