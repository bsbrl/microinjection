import cv2
import numpy as np
import glob
import random
from os import listdir
from os.path import isfile, join


# Load Yolo
net = cv2.dnn.readNet("ML_Yolo/yolov4-obj_last_miscroscope.weights", "ML_Yolo/yolov4_obj_microscope.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Name custom object
classes = ["pipe", "cell", "yolk"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
colors = np.array([[0.0,255.0,0.0], [0.0,0.0,255.0], [255.0, 0.0, 0.0]])

def YOLO_ML_2(img):
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    # print(outs)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    confidences_pipe = []
    confidences_cell = []
    confidences_yolk = []
    boxes = []
    boxes_pipe = []
    boxes_cell = []
    boxes_yolk = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                # print('Class_id = ', class_id, 'Confidence = ', confidence)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                if class_id == 0:
                    boxes_pipe.append([x, y, w, h])
                    confidences_pipe.append(float(confidence))
                if class_id == 1:
                    boxes_cell.append([x, y, w, h])
                    confidences_cell.append(float(confidence))
                if class_id == 2:
                    boxes_yolk.append([x, y, w, h])
                    confidences_yolk.append(float(confidence))
                class_ids.append(class_id)
    
    indexes_pipe = cv2.dnn.NMSBoxes(boxes_pipe, confidences_pipe, 0.1, 0.05)
    indexes_cell = cv2.dnn.NMSBoxes(boxes_cell, confidences_cell, 0.1, 0.05)
    indexes_yolk = cv2.dnn.NMSBoxes(boxes_yolk, confidences_yolk, 0.1, 0.05)
    # print('indexes = ', indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes_pipe)):
        if i in indexes_pipe:
            x, y, w, h = boxes_pipe[i]
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
    for i in range(len(boxes_cell)):
        if i in indexes_cell:
            x, y, w, h = boxes_cell[i]
            color = colors[1]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            cv2.circle(img, (center_x, center_y), radius=10, color=(255, 255, 255), thickness= -1)
            
    for i in range(len(boxes_yolk)):
        if i in indexes_yolk:
            x, y, w, h = boxes_yolk[i]
            color = colors[2]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            cv2.circle(img, (center_x, center_y), radius=10, color=(0, 0, 0), thickness= -1)

    # img = cv2.resize(img, None, fx = 1, fy = 1)
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)
    return img, boxes_pipe, boxes_cell, boxes_yolk
    