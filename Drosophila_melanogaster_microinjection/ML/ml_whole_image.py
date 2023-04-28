# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:18:19 2021

@author: enet-joshi317-admin
"""

# import numpy as np
#import os
import sys
# import tensorflow as tf
import time
from ML.rowcol_fun import rowcol_fun
#from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("..")
from object_detection.utils import ops as utils_ops
#from utils import label_map_util
#from utils import visualization_utils as vis_util
# if tf.__version__ < '1.10':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

def ml(model_name,image_location,ml_threshold,im_width,im_height,tf,np):

    start = time.time()    
    MODEL_NAME = model_name
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    
    # List of the strings that is used to add correct label for each box.
#    PATH_TO_LABELS = os.path.join(training_file, 'object-detection.pbtxt')
##    
#    NUM_CLASSES = 3
    
    # List of the strings that is used to add correct label for each box.
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')  
        
#    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#    category_index = label_map_util.create_category_index(categories)
    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    
    # In[46]:
    
    
    
    
    # ## Helper code
    
    # In[47]:
    
    
    def load_image_into_numpy_array(image,im_height,im_width):
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)
      
    # # Detection
    # In[50];
    def run_inference_for_multiple_images(images, graph):
      with graph.as_default():
        with tf.compat.v1.Session() as sess:
          output_dicts = []     
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
    output_dict_detection_scores_stored=[]
    output_dict_detection_boxes_stored=[]
    output_dict_detection_classes_stored=[]
    image = Image.open(image_location)
#    image_np = load_image_into_numpy_array(image,im_height,im_width)
    output_dicts = run_inference_for_multiple_images(image, detection_graph)
        
#    IMAGE_SIZE = (15, 15)
    xc_rc=[]
    yc_rc=[]
    scores=[]
    for r in range(len(output_dicts)):
        output_dict_detection_scores_stored.append(sorted([i for i in output_dicts[r]['detection_scores'] if i >= ml_threshold],reverse=True))
        output_dict_detection_boxes_stored.append(output_dicts[r]['detection_boxes'][0:len(output_dict_detection_scores_stored[r])])
        output_dict_detection_classes_stored.append(output_dicts[r]['detection_classes'][0:len(output_dict_detection_scores_stored[r])])
    print(output_dict_detection_scores_stored)
    for f in range(len(output_dict_detection_boxes_stored)):
        if len(output_dict_detection_boxes_stored[f])!=0:
            for i in range(len(output_dict_detection_boxes_stored[f])):
#                if output_dict_detection_classes_stored[f][i]==1 or output_dict_detection_classes_stored[f][i]==2:
                if output_dict_detection_classes_stored[f][i]==1:
                    y1=int(output_dict_detection_boxes_stored[f][i][0]*im_height)
                    y2=int(output_dict_detection_boxes_stored[f][i][2]*im_height)
                    x1=int(output_dict_detection_boxes_stored[f][i][1]*im_width)
                    x2=int(output_dict_detection_boxes_stored[f][i][3]*im_width)
                    r,c=rowcol_fun(x1,x2,y1,y2)
#                    y1a_rc.append([y1,r,c,1])
#                    y2a_rc.append([y2,r,c,1])
#                    x1a_rc.append([x1,r,c,1])
#                    x2a_rc.append([x2,r,c,1])
                    xc_rc.append([np.mean([x1,x2]),x1,x2,r,c])
                    yc_rc.append([np.mean([y1,y2]),y1,y2,r,c])
                    scores.append(output_dict_detection_scores_stored[f][i])
                else:
                    print('not single embryo')
#            vis_util.visualize_boxes_and_labels_on_image_array(
#                image_np,
#                output_dict_detection_boxes_stored[f],
#                output_dict_detection_classes_stored[f],
#                output_dict_detection_scores_stored[f],
#                category_index,
#                instance_masks=output_dicts[f].get('detection_masks'),
#                use_normalized_coordinates=True,
#                min_score_thresh=ml_threshold,
#                line_thickness=3)
#            plt.figure(figsize=IMAGE_SIZE)
#            plt.imshow(image_np)
        else:
            print('No embryos detected')   
#    for i in range(10):
#        for j in range(15):
#            for k in range(len(xc_rc)):
#                if i*400<yc_rc[k][0]<=(i+1)*400 and j*400<xc_rc[k][0]<=(j+1)*400:
#                    y1a_rc.append([int(yc_rc[k][1]),yc_rc[k][3],yc_rc[k][4],1])
#                    y2a_rc.append([int(yc_rc[k][2]),yc_rc[k][3],yc_rc[k][4],1])
#                    x1a_rc.append([int(xc_rc[k][1]),xc_rc[k][3],xc_rc[k][4],1])
#                    x2a_rc.append([int(xc_rc[k][2]),xc_rc[k][3],xc_rc[k][4],1])
    end = time.time()
        
    print('Time to run (s) = ',end-start)
    return xc_rc,yc_rc,scores       
# # from matplotlib import pyplot as plt
# import cv2
# from order import order 
# thresh_ml=.1
# all_scores=[]
# for k in range(10):
#     xc_rc,yc_rc,scores=ml('C:/Users/User/Downloads/Andrew_files/faster_r_cnn_trained_model_petri_new_8','C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_poster_dros_22_new/{}.jpg'.format(k+1),thresh_ml,6000,4000)
#     img_dish=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_poster_dros_22_new/{}.jpg'.format(k+1),1)
#     y1a_rc,y2a_rc,x1a_rc,x2a_rc=order(xc_rc,yc_rc,0,0)
#     all_scores.append(scores)
#     for i in range(len(y1a_rc)):
#         cv2.rectangle(img_dish,(int(xc_rc[i][0])-30,int(yc_rc[i][0])-30),(int(xc_rc[i][0])+30,int(yc_rc[i][0])+30),(0,255,0),5)
#         cv2.putText(img_dish, str(int(scores[i]*100))+'%'+'Embryo', (int(xc_rc[i][0])-60,int(yc_rc[i][0])-40), cv2.FONT_HERSHEY_SIMPLEX,.7, (0,0,0), 2, cv2.LINE_AA)
#         # cv2.putText(img_dish,'Embryo', (int(xc_rc[i][0])-60,int(yc_rc[i][0])-40), cv2.FONT_HERSHEY_SIMPLEX,.7, (0,0,0), 2, cv2.LINE_AA)
#     cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_poster_dros_22_new/{}_detected.jpg'.format(k+1),img_dish)
# # img_dish_new=cv2.resize(img_dish,(1600,1067))
# # y1a_rc,y2a_rc,x1a_rc,x2a_rc=order(xc_rc,yc_rc,0,0)
# # for i in range(len(y1a_rc)):
# #     cv2.rectangle(img_dish_new,(int(x1a_rc[i][0]*(float(1600)/float(6000))),int(y1a_rc[i][0]*(float(1067)/float(4000)))),(int(x2a_rc[i][0]*(float(1600)/float(6000))),int(y2a_rc[i][0]*(float(1067)/float(4000)))),(0,255,0),1)
# # cv2.rectangle(img_dish_new,(int(0*(float(1600)/float(6000))),int(0*(float(1067)/float(4000)))),(int(4080*(float(1600)/float(6000))),int(2602*(float(1067)/float(4000)))),(0,125,255),1)
# # cv2.imshow('Petri Dish ML Detections',img_dish_new)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
  
