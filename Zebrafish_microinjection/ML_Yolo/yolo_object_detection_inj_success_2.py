import cv2
import numpy as np
import glob
import random
from os import listdir
from os.path import isfile, join


# Load Yolo
net = cv2.dnn.readNet("ML_Yolo/yolov4-obj_inj_success.weights", "ML_Yolo/yolov4_obj_inj_success.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Name custom object
classes = ["Success", "Unsuccess"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
colors = np.array([[255], [0.0]])

def YOLO_inj_success_2(img):
    # Loading image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img_gray, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    # print(outs)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    confidences_succ = []
    confidences_unsu = []
    boxes = []
    boxes_succ = []
    boxes_unsu = []
    success_status = 0
    unsuccess_status = 0
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
                    boxes_succ.append([x, y, w, h])
                    confidences_succ.append(float(confidence))
                if class_id == 1:
                    boxes_unsu.append([x, y, w, h])
                    confidences_unsu.append(float(confidence))
                class_ids.append(class_id)
    
    indexes_succ = cv2.dnn.NMSBoxes(boxes_succ, confidences_succ, 0.1, 0.05)
    indexes_unsu = cv2.dnn.NMSBoxes(boxes_unsu, confidences_unsu, 0.1, 0.05)
    # print('indexes = ', indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes_succ)):
        if i in indexes_succ:
            x, y, w, h = boxes_succ[i]
            color = colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, 'Success', (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2, cv2.LINE_AA)
            
    for i in range(len(boxes_unsu)):
        if i in indexes_unsu:
            x, y, w, h = boxes_unsu[i]
            color = colors[1]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, 'Unsuccess', (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2, cv2.LINE_AA)

    # img = cv2.resize(img, None, fx = 1, fy = 1)
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)
    if boxes_succ:
        success_status = 1
    elif boxes_unsu:
        unsuccess_status = 1
    return img, success_status, unsuccess_status
    