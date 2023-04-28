import cv2
import numpy as np
import glob
import random
from os import listdir
from os.path import isfile, join
from ML_Yolo.row_col_list_fun_yolo import row_col_list_fun_yolo
from numba import vectorize, guvectorize
import time
import argparse

# Load Yolo
net = cv2.dnn.readNet("ML_Yolo/yolov3_training_last.weights", "ML_Yolo/yolov3_testing.cfg")
start_time = time.time()

# Name custom object
classes = ["yolk"]

# YOLO properties layer names defined 
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def YOLO_ML_1(img):
    # Loading image
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                # print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    
    # Draw Rectangle
    for i in range(len(boxes)):
        # print('boxes detected')
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            cv2.circle(img, (center_x, center_y), radius=10, color=(0, 0, 255), thickness= -1)
    
    
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)
    return img, boxes, indexes

# left_img = cv2.imread('image_captured1.jpg')
# right_img = cv2.imread('image_captured2.jpg')
# cur_frame_l, boxes_cur_l, indexes_cur_l = YOLO_ML(left_img)
# cur_frame_r, boxes_cur_r, indexes_cur_r = YOLO_ML(right_img)
# for i in range(len(boxes_cur_l)):
#         # print('boxes detected')
#         if i in indexes_cur_l:
#             x, y, w, h = boxes_cur_l[i]
#             cv2.rectangle(cur_frame_l, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
# for i in range(len(boxes_cur_r)):
#         # print('boxes detected')
#         if i in indexes_cur_r:
#             x, y, w, h = boxes_cur_r[i]
#             cv2.rectangle(cur_frame_r, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
# cv2.imwrite('left_output.jpg', cur_frame_l)
# cv2.imwrite('right_output.jpg', cur_frame_r)


# for frame_num in range(int(total_frame_num)):
# # for frame_num in range(3000):
#     ret, cur_frame = vid_in.read()
#     if frame_num % 7 == 0:
#         cur_frame, boxes_cur, indexes_cur = call_YOLO_ML_video(cur_frame)
#     for i in range(len(boxes_cur)):
#         # print('boxes detected')
#         if i in indexes_cur:
#             x, y, w, h = boxes_cur[i]
#             cv2.rectangle(cur_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     vid_out.write(cur_frame)
    

# vid_in.release()
# vid_out.release()
# print("--- %s seconds ---" % (time.time() - start_time))