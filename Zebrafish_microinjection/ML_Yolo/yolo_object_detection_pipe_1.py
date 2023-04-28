import cv2
import numpy as np
import glob
import random
from os import listdir
from os.path import isfile, join

# Load Yolo
net = cv2.dnn.readNet("ML_Yolo/yolov4-obj_best_pipe.weights", "ML_Yolo/yolov4-obj_pipe.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Name custom object
classes = ["Pipette"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
colors = np.array([[0.0,255.0,0.0], [0.0,0.0,255.0], [255.0, 0.0, 0.0]])

# loop through all the images
def YOLO_pipe_1(img):
    # Loading image
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
    
    indexes_pipe = cv2.dnn.NMSBoxes(boxes_pipe, confidences_pipe, 0.1, 0.05)

    # print('indexes = ', indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes_pipe)):
        if i in indexes_pipe:
            x, y, w, h = boxes_pipe[i]
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            center_x = int(x + w/2)
            center_y = int(y + h)
            cv2.circle(img, (center_x, center_y), radius=2, color=(0, 0, 255), thickness= -1)
            
    
    return img, boxes_pipe
    