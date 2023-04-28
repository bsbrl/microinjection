import cv2
import numpy as np
import glob
import random
from os import listdir
from os.path import isfile, join
from ML_Yolo.divide_image_yolo import divide_image_yolo
from ML_Yolo.row_col_list_fun import row_col_list_fun
# from divide_image_yolo import divide_image_yolo
# from row_col_list_fun import row_col_list_fun

def detections_dslr_divide_yolo(petri_name, path, ml_threashold):
    # Load Yolo
    net = cv2.dnn.readNet("ML_Yolo/yolov4-obj_last_DSLR_divide.weights", "ML_Yolo/yolov4_obj_DSLR_divide.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # net = cv2.dnn.readNet("yolov4-obj_last_DSLR_divide.weights", "yolov4_obj_DSLR_divide.cfg")
    
    # Name custom object
    classes = ["single_embryo", "dead_embryo", "bubble"]
    
    # Read single image
    print(path+petri_name)
    ori_img = cv2.imread(path+petri_name)
    
    # Divide image
    divide_image_yolo(6000, 4000, petri_name, 1, path)
    
    # Images path
    images_path_test = "D:\Microinjection_Project\Python_Code\ML_Yolo\divided_img"
    images_path = glob.glob(images_path_test+"\*.jpg")
    
    # Get rows and column of each images
    onlyfiles = [f for f in listdir(images_path_test) if isfile(join(images_path_test, f))]
    row,col=row_col_list_fun(onlyfiles)
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors = np.array([[0.0,255.0,0.0], [0.0,0.0,255.0], [125.0, 125.0, 0.0]])
    
    # Insert here the path of your images
    # random.shuffle(images_path)
    img_num = 0
    out_image = np.zeros((4000,6000,3), np.uint8)
    box_center = []
    all_class_ids = []
    all_box_coordinate = []
    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape
    
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        net.setInput(blob)
        outs = net.forward(output_layers)
        # print(outs)
    
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    # print('Class_id = ', class_id)
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
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.05)
        # print('indexes = ', indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
                # Center co-ordinates
                center = [x+(w/2) + ((col[img_num]-1)*width), y+(h/2) + ((row[img_num]-1)*height)]
                box_coordinate = [y + ((row[img_num]-1)*height), x + ((col[img_num]-1)*width), (y + h) + ((row[img_num]-1)*height), (x + w) + ((col[img_num]-1)*width)]
                all_class_ids.append(class_ids[i])
                all_box_coordinate.append(box_coordinate)
                box_center.append(center)
                # cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
    
        img = cv2.resize(img, None, fx = 1, fy = 1)
        # cv2.imshow("Image", img)
        # key = cv2.waitKey(0)
        out_image[(row[img_num]-1)*height:(row[img_num])*height, (col[img_num]-1)*width:(col[img_num])*width] = img
        img_num = img_num + 1
    
    box_center = np.asarray(box_center)
    all_class_ids = np.asarray(all_class_ids)
    all_box_coordinate = np.asarray(all_box_coordinate)
    np.save('box_coordinate.npy', all_box_coordinate)
    np.save('box_center.npy', box_center)
    np.save('class_ids.npy', all_class_ids)
    cv2.imwrite('DSLR_image_detection.jpg', out_image)
    
# detections_dslr_divide_yolo('DSLR_image_1.jpg', 'D:/Microinjection_Project/Python_Code/', 50)