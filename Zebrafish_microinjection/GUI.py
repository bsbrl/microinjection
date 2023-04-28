# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:05:36 2020

@author: Amey Joshi
"""


import tkinter as tk
from tkinter import messagebox as tkMessageBox
from PIL import ImageTk, Image
import threading
from queue import Queue
import serial
import time
import datetime
import numpy as np
import math
import pyautogui
from XYZ_stage.MyXYZ import MyXYZ
from DSLR_Camera.DSLR_Call import func_TakeNikonPicture
from MachineLearning_Zebrafish.detection_ML_forGUI import detections_dslr_image
from CameraWorkbench_master.camera import *
import os
from os import path
# from LED.LED_onoff import LED
from LED.LEDs_Both_onoff import LED
from Sensapex_Manipulator.MyUMP import MyUMP
from Transformation_matrix.transformation_DSLR_4x import transformation_DSLR_4x
from Transformation_matrix.transformation_DSLR_4x import transformation_DSLR_inj
from Transformation_matrix.transformation_inj_pip import transformation_inj_pip
from Transformation_matrix.transformation_inj_pip import transformation_pip_z
from Transformation_matrix.transformation_vial import transformation_vial
from Autofocus.autofocus import autofocus
from Needle_detection.needle_detection import needle_detection
from Needle_detection.needle_detection_live_1 import needle_detection_live_1
from Needle_detection.needle_detection_live_2 import needle_detection_live_2
from Pressure_system.MyPressureSystem import MyPressureSystem
from Path_planning.My_path_planning import My_path_planning
from ML_Yolo.yolo_object_detection_microscope_1 import YOLO_ML_1
from ML_Yolo.yolo_object_detection_microscope_2 import YOLO_ML_2
from ML_Yolo.yolo_object_detection_pipe_1 import YOLO_pipe_1
from ML_Yolo.yolo_object_detection_pipe_2 import YOLO_pipe_2
from ML_Yolo.yolo_object_detection_inj_success_1 import YOLO_inj_success_1
from ML_Yolo.yolo_object_detection_inj_success_2 import YOLO_inj_success_2
from ML_Yolo.yolo_object_detection_DSLR_divide import detections_dslr_divide_yolo

ML_flag = 0
app = tk.Tk()
app.geometry("1920x1080+0+0")
os.chdir('D:/Microinjection_Project/Python_Code/')

def DSLR_image(input_filename, Dish_number):
    print('Start taking DSLR image')
    start_time = time.time()
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), 0)
    if Dish_number == 1:
        values[0].delete(0, tk.END)
        values[0].insert(10, default_DSLR_1[0])
        values[1].delete(0, tk.END)
        values[1].insert(10, default_DSLR_1[1])
        values[2].delete(0, tk.END)
        values[2].insert(10, default_DSLR_1[2])
    if Dish_number == 2:
        values[0].delete(0, tk.END)
        values[0].insert(10, default_DSLR_2[0])
        values[1].delete(0, tk.END)
        values[1].insert(10, default_DSLR_2[1])
        values[2].delete(0, tk.END)
        values[2].insert(10, default_DSLR_2[2])
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), 0)
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    func_TakeNikonPicture(input_filename, 'D:/Microinjection_Project/Python_Code/')
    # time.sleep(2)
    image = cv2.imread(input_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 427))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(image=image)
    panel.image = image
    if Dish_number == 1:
        panel.place(x=475, y=25)
        x_place = 475
        y_place = 25
    if Dish_number == 2:
        panel.place(x=1250, y=25)
        x_place = 1250
        y_place = 25
    panel = tk.Label(app, text='DSLR {} time = {} sec'.format(Dish_number, round(time.time() - start_time, 2)))
    panel.place(x=x_place+50, y=y_place+450)
    print('Time = ', time.time() - start_time, 'sec')
    print('DSLR image taken')
    
    
def ML_detect_faster_RCNN(input_filename, path, dish_number):
    if dish_number == 1:
        x_place = 475
        y_place = 25
    if dish_number == 2:
        x_place = 1250
        y_place = 25
    detections_dslr_image(input_filename, path, float((values[6]).get()))
    image = cv2.imread('DSLR_image_detection.jpg')
    cv2.imwrite('DSLR_image_detection_{}.jpg'.format(dish_number), image)
    cv2.imwrite('DSLR_image_with_clicked_Box_{}.jpg'.format(dish_number), image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 427))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(image=image)
    panel.image = image
    panel.place(x=x_place, y=y_place)
    output_dict_detection_classes_stored = np.load('output_dict_detection_classes_stored.npy', allow_pickle=True)
    num_1 = 0
    num_2 = 0
    num_3 = 0
    for i in range(len(output_dict_detection_classes_stored)):
        num_1 = np.count_nonzero(output_dict_detection_classes_stored[i] == 1) + num_1
        num_2 = np.count_nonzero(output_dict_detection_classes_stored[i] == 2) + num_2
        num_3 = np.count_nonzero(output_dict_detection_classes_stored[i] == 3) + num_3
    panel = tk.Label(app, text='Alive Embryo Detected = {}'.format(num_1))
    panel.place(x=x_place+50, y=y_place+430)
    panel = tk.Label(app, text='Dead Embryo Detected = {}'.format(num_2))
    panel.place(x=x_place+250, y=y_place+430)
    panel = tk.Label(app, text='Bubbles Detected = {}'.format(num_3))
    panel.place(x=x_place+450, y=y_place+430)
    
def ML_detect_YOLO(input_filename, path, dish_number):
    print('Yolo detection start')
    start_time = time.time()
    if dish_number == 1:
        x_place = 475
        y_place = 25
    if dish_number == 2:
        x_place = 1250
        y_place = 25
    detections_dslr_divide_yolo(input_filename, path, float((values[6]).get()))
    image = cv2.imread('DSLR_image_detection.jpg')
    cv2.imwrite('DSLR_image_detection_{}.jpg'.format(dish_number), image)
    cv2.imwrite('DSLR_image_with_clicked_Box_{}.jpg'.format(dish_number), image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 427))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(image=image)
    panel.image = image
    panel.place(x=x_place, y=y_place)
    class_ids = np.load('class_ids.npy', allow_pickle=True)
    Alive_num = np.count_nonzero(class_ids == 0)
    Dead_num = np.count_nonzero(class_ids == 1)
    Bubble_num = np.count_nonzero(class_ids == 2)
    panel = tk.Label(app, text='Alive Embryo Detected = {}'.format(Alive_num))
    panel.place(x=x_place+50, y=y_place+430)
    panel = tk.Label(app, text='Dead Embryo Detected = {}'.format(Dead_num))
    panel.place(x=x_place+250, y=y_place+430)
    panel = tk.Label(app, text='Bubbles Detected = {}'.format(Bubble_num))
    panel.place(x=x_place+450, y=y_place+430)
    panel = tk.Label(app, text='Yolo detection {} time = {} sec'.format(dish_number, round(time.time() - start_time, 2)))
    panel.place(x=x_place+250, y=y_place+450)
    print('Time = ', time.time() - start_time, 'sec')
    print('Yolo detection finished')
    
def call_ml_yolo(values):
    yolo_detection_start[0] = True
    b54.configure(bg='green')    
    
def go_to_pos_GUI(values):
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
   
def XYZ_set_speed(values):
    XYZ = MyXYZ()
    XYZ.set_Velocity(float((values[18]).get()), float((values[19]).get()), float((values[20]).get()))
   
def select_XYZ():
    XYZ = MyXYZ()

def camera_on(videoloop_stop):
    threading.Thread(target=videoLoop, args=(videoloop_stop,)).start()
    print('Camera is on')
    b5.configure(bg='green')
    b41.configure(bg=app.cget('bg'))
    
def camera_off(videoloop_stop):
    videoloop_stop[0] = True
    b5.configure(bg=app.cget('bg'))
    b41.configure(bg='red')
    print('Camera is off')
    
def videoLoop(mirror=False):
    No = 0
    S=AmscopeCamera(0)
    S.activate()
    time.sleep(2)
    flag = 0
    panel = tk.Label(image=None)
    panel.image = None
    panel.place(x=1250, y=25)
    while True:
        frame=S.get_frame()
        time.sleep(0.01)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame=cv2.resize(frame, (640, 480))
        if mirror is True:
            frame = frame[:, ::-1]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel.configure(image=image)
        panel.image = image

        # check switcher value
        if autofocus_start[0]:
            autofocus_start[0] = False
            S.deactivate()
            panel.destroy()
            print('Camera is off')
            z_ref = autofocus(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
            values[2].delete(0, tk.END)
            values[2].insert(10, z_ref)
            videoloop_stop[0] = False
            camera_on(videoloop_stop)
            break
                
        if videoloop_stop[0]:
            # if switcher tells to stop then we switch it again and stop videoloop
            videoloop_stop[0] = False
            S.deactivate()
            panel.destroy()
            break
    return panel

def camera_on_inclined_1(videoloop_stop_inclined_1):
    threading.Thread(target=videoLoop_inclined_1, args=(videoloop_stop_inclined_1, 1)).start()
    print('Inclined Camera is on')
    b43.configure(bg='green')
    b44.configure(bg=app.cget('bg'))
    # b53.configure(bg=app.cget('bg'))
    
def camera_off_inclined_1(videoloop_stop_inclined_1):
    videoloop_stop_inclined_1[0] = True
    needle_detection_start_1[0] = False
    b43.configure(bg=app.cget('bg'))
    b44.configure(bg='red')
    print('Inclined Camera is off')
    
def camera_on_inclined_2(videoloop_stop_inclined_2):
    threading.Thread(target=videoLoop_inclined_2, args=(videoloop_stop_inclined_2, 2)).start()
    print('Inclined Camera is on')
    b51.configure(bg='green')
    b52.configure(bg=app.cget('bg'))
    # b46.configure(bg=app.cget('bg'))
    
def camera_off_inclined_2(videoloop_stop_inclined_2):
    videoloop_stop_inclined_2[0] = True
    b51.configure(bg=app.cget('bg'))
    b52.configure(bg='red')
    print('Inclined Camera is off')

def videoLoop_inclined_1(yolk_left, pipe_left, mirror=False):
    No = 0
    panels = []
    cap1 = cv2.VideoCapture(2)
    if not(cap1.isOpened()):
        print('Could not open video device 1')
    else:
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*1)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*1)
        flag = 0
        panel = tk.Label(image=None)
        panel.image = None
        panel.place(x=475, y=590)
        # panel_ned_pos = tk.Label(app, text='Neddle Detection = (0,0)')
        # panel_ned_pos.place(x=475, y=590)
        while True:
            # cap1 = cv2.VideoCapture(3)
            ret1, frame1 = cap1.read()
            # time.sleep(0.01)
            if needle_detection_start_1[0]:
                needle_detection_start_1[0] = False
                frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
                cv2.imwrite('Needle_image.jpg', frame1)
                # print('Inclined Camera is off')
                (x_needle, y_needle) = needle_detection(0)
                time.sleep(2)
                # needle_point_detected[0] = True
                # print(needle_point_detected)
                videoloop_stop_inclined_1[0] = True
                # cap1.release()
                # panel.destroy()
                image = cv2.imread('Needle_image_detected.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.rotate(image, cv2.ROTATE_180)
                image = cv2.resize(image, (640, 320))
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                panel_ned = tk.Label(image=image)
                panel_ned.image = image
                panel_ned.place(x=475, y=590)
                # panel_ned_pos = tk.Label(app, text='Neddle Detection = ({},{})'.format(x_needle,y_needle))
                # panel_ned_pos.place(x=475, y=590)
                print('Needle Detection Done')
                needle_detection_start_1[0] = False
                break
            
            if take_image_flag_2[0]:
                cv2.imwrite('image.jpg', frame2)
                take_image_flag_2[0] = True
            
            if yolo_detection_start[0]:
                # start_detection_time_1 = time.time()
                frame1, boxes_pipe, boxes_cell, boxes_yolk = YOLO_ML_1(frame1)
                frame1, boxes_pipe_tip = YOLO_pipe_1(frame1)
                # print('Detection time 1', time.time() - start_detection_time_1)
                if whole_start[0]:
                    yolk_left.put(boxes_cell)
                    pipe_left.put(boxes_pipe_tip)
                
            
            # # Live Needle detection code
            # frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
            # (x_needle, y_needle, frame1) = needle_detection_live_1(frame1)
            # panel_ned_pos.configure(text='Neddle Detection = ({},{})'.format(x_needle,y_needle))
            # panel_ned_pos.text = 'Neddle Detection = ({},{})'.format(x_needle,y_needle)
            # frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
            # # Live needle detection end 
            
            # Circles for references
            frame1 = cv2.circle(frame1, (0,360), radius=5, color=(0, 0, 255), thickness=-1)
            frame1 = cv2.circle(frame1, (1280,360), radius=5, color=(0, 0, 255), thickness=-1)
            frame1 = cv2.circle(frame1, (640,0), radius=5, color=(0, 0, 255), thickness=-1)
            frame1 = cv2.circle(frame1, (640,720), radius=5, color=(0, 0, 255), thickness=-1)
            frame1 = cv2.line(frame1, (0, 360), (1280, 360), (0, 0, 255), 1)
            frame1 = cv2.line(frame1, (640, 0), (640, 720), (0, 0, 255), 1)
            # frame1 = cv2.circle(frame1, (625,346), radius=5, color=(0, 0, 255), thickness=-1)
            
            frame1 = cv2.resize(frame1, (640, 320))
            if mirror is True:
                frame1 = frame1[:, ::-1]                
            image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            panel.configure(image=image)
            panel.image = image
            # check switcher value
            if videoloop_stop_inclined_1[0]:
                # if switcher tells to stop then we switch it again and stop videoloop
                videoloop_stop_inclined_1[0] = False
                cap1.release()
                app.update()
                time.sleep(0.01)
                panel.destroy()
                break                    
        return panel
    
def videoLoop_inclined_2(yolk_right, pipe_right, mirror=False):
    No = 0
    cap2 = cv2.VideoCapture(3)
    if not(cap2.isOpened()):
        print('Could not open video device')
    else:
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*1)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720*1)
        flag = 0
        panel = tk.Label(image=None)
        panel.image = None
        panel.place(x=1250, y=590)
        # panel_ned_pos = tk.Label(app, text='Neddle Detection = (0,0)')
        # panel_ned_pos.place(x=1250, y=590)
        while True:
            # cap = cv2.VideoCapture(4)
            ret2, frame2 = cap2.read()
            # time.sleep(0.01)
            if needle_detection_start_2[0]:
                needle_detection_start_2[0] = False
                frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
                cv2.imwrite('Needle_image.jpg', frame2)
                # print('Inclined Camera is off')
                (x_needle, y_needle) = needle_detection(0)
                time.sleep(2)
                # needle_point_detected[0] = True
                # print(needle_point_detected)
                videoloop_stop_inclined_2[0] = False
                # cap2.release()
                # panel.destroy()
                image = cv2.imread('Needle_image_detected.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.rotate(image, cv2.ROTATE_180)
                image = cv2.resize(image, (640, 320))
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                panel = tk.Label(image=image)
                panel.image = image
                panel.place(x=1250, y=590)
                # panel = tk.Label(app, text='Neddle Detection = ({},{})'.format(x_needle,y_needle))
                # panel.place(x=1250, y=590)
                print('Needle Detection Done')
                break
            
            if take_image_flag_2[0]:
                cv2.imwrite('image.jpg', frame2)
                take_image_flag_2[0] = True
            
            if yolo_detection_start[0]:
                # start_detection_time_2 = time.time()
                frame2, boxes_pipe, boxes_cell, boxes_yolk = YOLO_ML_2(frame2)
                frame2, boxes_pipe_tip = YOLO_pipe_2(frame2)
                # print('Detection time 2', time.time() - start_detection_time_2)
                if whole_start[0]:
                    yolk_right.put(boxes_cell)
                    pipe_right.put(boxes_pipe_tip)
                    
            
            # Live Needle detection code
            # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
            # (x_needle, y_needle, frame2) = needle_detection_live_2(frame2)
            # panel_ned_pos.configure(text='Neddle Detection = ({},{})'.format(x_needle,y_needle))
            # panel_ned_pos.text = 'Neddle Detection = ({},{})'.format(x_needle,y_needle)
            # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
            # Live needle detection end
            
            # frame2 =cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
            
            # Circles for references
            frame2 = cv2.circle(frame2, (0,360), radius=5, color=(0, 0, 255), thickness=-1)
            frame2 = cv2.circle(frame2, (1280,360), radius=5, color=(0, 0, 255), thickness=-1)
            frame2 = cv2.circle(frame2, (640,0), radius=5, color=(0, 0, 255), thickness=-1)
            frame2 = cv2.circle(frame2, (640,720), radius=5, color=(0, 0, 255), thickness=-1)
            frame2 = cv2.line(frame2, (0, 360), (1280, 360), (0, 0, 255), 1)
            frame2 = cv2.line(frame2, (640, 0), (640, 720), (0, 0, 255), 1)
            # frame2 = cv2.circle(frame2, (782,376), radius=5, color=(0, 0, 255), thickness=-1)
            
            frame2 = cv2.resize(frame2, (640, 320))
            if mirror is True:
                frame2 = frame2[:, ::-1]                
            image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            panel.configure(image=image)
            panel.image = image
            
            # check switcher value
            if videoloop_stop_inclined_2[0]:
                # if switcher tells to stop then we switch it again and stop videoloop
                videoloop_stop_inclined_2[0] = False
                cap2.release()
                app.update()
                time.sleep(0.01)
                panel.destroy()
                break                    
        return panel

def call_autofocus(values):
    autofocus_start[0] = True

def move_center(default_center, values):
    values[0].delete(0, tk.END)
    values[0].insert(10, default_center[0])
    values[1].delete(0, tk.END)
    values[1].insert(10, default_center[1])
    values[2].delete(0, tk.END)
    values[2].insert(10, default_center[2])
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    
def move_y_plus(values):
    m = float((values[1]).get()) - float((values[4]).get())
    values[1].delete(0, tk.END)
    values[1].insert(10, m)
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    
def move_y_negative(values):
    m = float((values[1]).get()) + float((values[4]).get())
    values[1].delete(0, tk.END)
    values[1].insert(10, m)
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    
def move_x_plus(values):
    m = float((values[0]).get()) - float((values[3]).get())
    values[0].delete(0, tk.END)
    values[0].insert(10, m)
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    
def move_x_negative(values):
    m = float((values[0]).get()) + float((values[3]).get())
    values[0].delete(0, tk.END)
    values[0].insert(10, m)
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    
def move_z_plus(values):
    m = float((values[2]).get()) + float((values[5]).get())
    values[2].delete(0, tk.END)
    values[2].insert(10, m)
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    
def move_z_negative(values):
    m = float((values[2]).get()) - float((values[5]).get())
    values[2].delete(0, tk.END)
    values[2].insert(10, m)
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    
def change_dx(value):
    values[3].delete(0, tk.END)
    values[3].insert(10, value)
    values[4].delete(0, tk.END)
    values[4].insert(10, value)
    values[5].delete(0, tk.END)
    values[5].insert(10, value)
    
def LED_on_off(value):
    if value == 0:
        b56.configure(bg='red')
        b18.configure(bg='red')
        b19.configure(bg='red')
        b55.configure(bg=app.cget('bg'))
    if value == 1:
        b18.configure(bg='green')
        b19.configure(bg='red')
        b55.configure(bg=app.cget('bg'))
        b56.configure(bg=app.cget('bg'))
    if value == 2:
        b18.configure(bg='red')
        b19.configure(bg='green')
        b55.configure(bg=app.cget('bg'))
        b56.configure(bg=app.cget('bg'))
    if value == 3:
        b55.configure(bg='green')
        b18.configure(bg='green')
        b19.configure(bg='green')
        b56.configure(bg=app.cget('bg'))
    LED(value)
    
def select_UMP():
    UMP = MyUMP()
    
def calibrate_UMP():
    MyUMP.Calibration(True)
    
def go_to_UMP_pos_GUI(values):
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))

def move_center_UMP(default_center, values):
    values[7].delete(0, tk.END)
    values[7].insert(10, default_center_UMP[0])
    values[8].delete(0, tk.END)
    values[8].insert(10, default_center_UMP[1])
    values[9].delete(0, tk.END)
    values[9].insert(10, default_center_UMP[2])
    values[10].delete(0, tk.END)
    values[10].insert(10, default_center_UMP[3])
    values[11].delete(0, tk.END)
    values[11].insert(10, default_center_UMP[4])
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))

def move_x_plus_UMP(values):
    m = float((values[7]).get()) + float((values[12]).get())
    values[7].delete(0, tk.END)
    values[7].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    

def move_x_negative_UMP(values):
    m = float((values[7]).get()) - float((values[12]).get())
    values[7].delete(0, tk.END)
    values[7].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    
    
def move_y_plus_UMP(values):
    m = float((values[8]).get()) + float((values[13]).get())
    values[8].delete(0, tk.END)
    values[8].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    

def move_y_negative_UMP(values):
    m = float((values[8]).get()) - float((values[13]).get())
    values[8].delete(0, tk.END)
    values[8].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    

def move_z_plus_UMP(values):
    m = float((values[9]).get()) + float((values[14]).get())
    values[9].delete(0, tk.END)
    values[9].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    

def move_z_negative_UMP(values):
    m = float((values[9]).get()) - float((values[14]).get())
    values[9].delete(0, tk.END)
    values[9].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    
    
def move_d_plus_UMP(values):
    m = float((values[10]).get()) + float((values[15]).get())
    values[10].delete(0, tk.END)
    values[10].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    

def move_d_negative_UMP(values):
    m = float((values[10]).get()) - float((values[15]).get())
    values[10].delete(0, tk.END)
    values[10].insert(10, m)
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))    

def change_dx_UMP(value):
    values[12].delete(0, tk.END)
    values[12].insert(10, value)
    values[13].delete(0, tk.END)
    values[13].insert(10, value)
    values[14].delete(0, tk.END)
    values[14].insert(10, value)
    values[15].delete(0, tk.END)
    values[15].insert(10, value)    

def change_speed_UMP(value):
    values[11].delete(0, tk.END)
    values[11].insert(10, value)  
    
def getorigin(eventorigin):
    x = eventorigin.x
    y = eventorigin.y
    x = 8*x
    y = 8*y
    print(x, y)
    box_center = np.load('box_center.npy')
    dis = np.zeros((len(box_center),1))
    for i in range(len(box_center)):
        dis[i][0] = np.square(x - (box_center[i][0])) + np.square(y - (box_center[i][1]))
    index = np.argmin(dis)
    box_coordinate = np.load('output_dict_detection_boxes_stored_modi.npy')
    box_center_x = box_center[index][0]#*6000
    box_center_y = box_center[index][1]#*4000
    # print(box_center[index][0]*6000, box_center[index][1]*4000)
    image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_detection.jpg')
    image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_with_clicked_Box.jpg')
    # start_point = (int(box_coordinate[0][index][1]*6000), int(box_coordinate[0][index][0]*4000))
    # end_point = (int(box_coordinate[0][index][3]*6000), int(box_coordinate[0][index][2]*4000))
    start_point = (int(box_coordinate[index][1]), int(box_coordinate[index][0]))
    end_point = (int(box_coordinate[index][3]), int(box_coordinate[index][2]))
    image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 12)
    # image = cv2.circle(image, (box_center_x, box_center_y), 50, (255, 0, 0), 2)
    image_record = cv2.rectangle(image, start_point, end_point, (255, 255, 0), 12)
    cv2.imwrite('DSLR_image_with_clicked_Box.jpg', image_record)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 427))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(image=image)
    panel.image = image
    panel.place(x=475, y=25)
    (x_stage, y_stage) = transformation_DSLR_4x(box_center_x, box_center_y)
    XYZ = MyXYZ()
    # Comment out this if you want to go to 4x objective
    # XYZ.Position(x_stage, y_stage, z_reference)
    values[0].delete(0, tk.END)
    values[0].insert(10, x_stage)
    values[1].delete(0, tk.END)
    values[1].insert(10, y_stage)
    values[2].delete(0, tk.END)
    values[2].insert(10, z_reference)
    
def needle_position():
    values[0].delete(0, tk.END)
    values[0].insert(10, default_needle_XYZ[0])
    values[1].delete(0, tk.END)
    values[1].insert(10, default_needle_XYZ[1])
    values[2].delete(0, tk.END)
    values[2].insert(10, default_needle_XYZ[2])
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()), float((values[1]).get()), float((values[2]).get()))
    values[7].delete(0, tk.END)
    values[7].insert(10, default_needle[0])
    values[8].delete(0, tk.END)
    values[8].insert(10, default_needle[1])
    values[9].delete(0, tk.END)
    values[9].insert(10, default_needle[2])
    values[10].delete(0, tk.END)
    values[10].insert(10, default_needle[3])
    values[11].delete(0, tk.END)
    values[11].insert(10, default_needle[4])
    MyUMP.Position(float((values[7]).get()), float((values[8]).get()), float((values[9]).get()), float((values[10]).get()), float((values[11]).get()))
    
def get_inject_status(status):
    if status == True:
        inject_status[0] = True
        b48.configure(bg='green')
        b49.configure(bg='red')
    if status == False:
        inject_status[0] = False
        b48.configure(bg='red')
        b49.configure(bg='green')

def inject_fun(values):
    if inject_status[0] == True:
        call_pressure(values, 'I')
    if inject_status[0] == False:
        call_pressure(values, 'W')
        

def call_needle_detection_1(values):
    needle_detection_start_1[0] = True
    b43.configure(bg=app.cget('bg'))
    b44.configure(bg='red')
    b53.configure(bg='green')
    print('Inclined Camera is off')
    
def call_needle_detection_2(values):
    needle_detection_start_2[0] = True
    b51.configure(bg=app.cget('bg'))
    b52.configure(bg='red')
    b46.configure(bg='green')
    print('Inclined Camera is off')
    
def call_inject(values):
    XYZ = MyXYZ()
    XYZ.Position(float((values[0]).get()) + default_inject[0], float((values[1]).get()) + default_inject[1], 15)
    # time.sleep(1)
    XYZ.Position(float((values[0]).get()) + default_inject[0], float((values[1]).get()) + default_inject[1], float((values[2]).get()) + default_inject[2] - 1)
    # time.sleep(1)
    XYZ.Position(float((values[0]).get()) + default_inject[0], float((values[1]).get()) + default_inject[1], float((values[2]).get()) + default_inject[2] - 0.5)
    # time.sleep(1)
    XYZ.Position(float((values[0]).get()) + default_inject[0], float((values[1]).get()) + default_inject[1], float((values[2]).get()) + default_inject[2])
    time.sleep(1)
    call_pressure(values, 'I')
    time.sleep(1)
    XYZ.Position(float((values[0]).get()) + default_inject[0], float((values[1]).get()) + default_inject[1], 15)
    # m = float((values[0]).get()) - 80.2284
    # n = float((values[1]).get()) + 2.1979
    # p = float((values[2]).get()) + 16.8
    m = float((values[0]).get()) + default_inject[0]
    n = float((values[1]).get()) + default_inject[1]
    # p = float((values[2]).get()) + 15
    p = 15
    values[0].delete(0, tk.END)
    values[0].insert(10, m)
    values[1].delete(0, tk.END)
    values[1].insert(10, n)
    values[2].delete(0, tk.END)
    values[2].insert(10, p)

def recording(mirror=False):
    # display screen resolution, get it from your OS settings
    SCREEN_SIZE = (1920, 1080)
    # define the codec
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # create the video write object
    out = cv2.VideoWriter("output.avi", fourcc, 60.0, (SCREEN_SIZE))
    while True:
        # make a screenshot
        img = pyautogui.screenshot()
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        # convert colors from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # write the frame
        out.write(frame)
        # if the user clicks q, it exits        
        if recording_stop[0] == True:
            cv2.destroyAllWindows()
            out.release()
            break
    
def stop_recording(mirror = False):
    recording_stop = [True]
    
def Data_save(dish_number, injection_material, values, current_time, unsuccess_inj, success_inj):
    time = datetime.datetime.now()
    time = time.strftime("%m-%d-%Y_%H-%M-%S")
    os.mkdir('D:/Microinjection_Project/Python_Code/Injection_Data/'+time)
    ori_img = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_{}.jpg'.format(dish_number))
    yolo_img = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_detection_{}.jpg'.format(dish_number))
    inje_img = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_with_clicked_Box_{}.jpg'.format(dish_number))
    path_img = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_2opt_path.jpg')
    box_center = np.load('D:/Microinjection_Project/Python_Code/box_center.npy')
    box_coordinate = np.load('D:/Microinjection_Project/Python_Code/box_coordinate.npy')
    class_ids = np.load('D:/Microinjection_Project/Python_Code/class_ids.npy')
    
    file = open("D:/Microinjection_Project/Python_Code/Injection_Data/{}/Injection_{}.txt".format(time, time), "w+")
    file.write("Date and Time: %s \n" %(time))
    file.write("Injection material: %s \n" %(injection_material[dish_number-1]))
    file.write("Volume: %d nl\n" %(int((values[16]).get())))
    file.write("Rate: %d nl/sec \n" %(int((values[17]).get())))
    Alive_num = np.count_nonzero(class_ids == 0)
    Dead_num = np.count_nonzero(class_ids == 1)
    Bubble_num = np.count_nonzero(class_ids == 2)
    file.write("Alive embryo: %d Dead embryo: %d Bubble: %d \n" %(Alive_num, Dead_num, Bubble_num))
    file.write("Successful injection: %d Unsuccessful injection: %d \n" %(success_inj, unsuccess_inj))
    file.write("Injection time: %s min \n" %(str(current_time)))
    file.close()
    
    cv2.imwrite('D:/Microinjection_Project/Python_Code/Injection_Data/{}/DSLR_image_{}.jpg'.format(time, dish_number), ori_img)
    cv2.imwrite('D:/Microinjection_Project/Python_Code/Injection_Data/{}/DSLR_image_detection_{}.jpg'.format(time, dish_number), yolo_img)
    cv2.imwrite('D:/Microinjection_Project/Python_Code/Injection_Data/{}/DSLR_image_with_clicked_Box_{}.jpg'.format(time, dish_number), inje_img)
    cv2.imwrite('D:/Microinjection_Project/Python_Code/Injection_Data/{}/DSLR_image_2opt_path.jpg'.format(time), path_img)
    np.save('D:/Microinjection_Project/Python_Code/Injection_Data/{}/box_center.npy'.format(time), box_center)
    np.save('D:/Microinjection_Project/Python_Code/Injection_Data/{}/box_coordinate.npy'.format(time), box_coordinate)
    np.save('D:/Microinjection_Project/Python_Code/Injection_Data/{}/class_ids.npy'.format(time), class_ids)
    
def call_pressure(values, direction):
    Pressure = MyPressureSystem()
    if direction == 'I':
        Pressure.inject(float((values[16]).get()), float((values[17]).get()))
    if direction == 'W':
        Pressure.withdraw(float((values[16]).get()), float((values[17]).get()))
        
def vial(vial_num, volume, rate, status):
    XYZ = MyXYZ()
    Position = XYZ.Get_Pos()
    XYZ.Position(Position['1'], Position['2'], 0)
    x, y, z = transformation_vial(vial_num)
    XYZ.Position(x, y, 0)
    XYZ.Position(x, y, z)
    # Add pressure line here
    Pressure = MyPressureSystem()
    if status[0] == True:
        Pressure.inject(volume, rate)
    if status[0] == False:
        Pressure.withdraw(volume, rate)
    time.sleep(2)
    XYZ.Position(x, y, 0)
    values[0].delete(0, tk.END)
    values[0].insert(10, x)
    values[1].delete(0, tk.END)
    values[1].insert(10, y)
    values[2].delete(0, tk.END)
    values[2].insert(10, 0)
    
def start_whole(mirror=False):
    LED_on_off(3)
    # DSLR_image('DSLR_image.jpg')
    app.update()
    ML_detect_faster_RCNN('DSLR_image.jpg', 'D:/Microinjection_Project/Python_Code/', 1)
    app.update()
    box_center = np.load('box_center.npy')
    box_coordinate = np.load('output_dict_detection_boxes_stored_modi.npy')
    number_detection = len(box_center)
    pre_box_center_x = int(box_center[0][0])
    pre_box_center_y = int(box_center[0][1])
    for i in range(number_detection):
        box_center_x = box_center[i][0]
        box_center_y = box_center[i][1]
        (x_stage, y_stage) = transformation_DSLR_4x(box_center_x, box_center_y)
        # image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_detection.jpg')
        image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_with_clicked_Box.jpg')
        start_point = (int(box_coordinate[i][1]), int(box_coordinate[i][0]))
        end_point = (int(box_coordinate[i][3]), int(box_coordinate[i][2]))
        image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 12)
        image = cv2.line(image, (int(pre_box_center_x), int(pre_box_center_y)), (int(box_center_x), int(box_center_y)), (0, 0, 255), 3)
        image_record = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 427))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel = tk.Label(image=image)
        panel.image = image
        panel.place(x=475, y=25)
        if i == 0:
            image_record = cv2.rectangle(image_record, start_point, end_point, (0, 0, 0), 12)
        elif i == (number_detection - 1):
            image_record = cv2.rectangle(image_record, start_point, end_point, (255, 0, 0), 12)
        else:
            image_record = cv2.rectangle(image_record, start_point, end_point, (50, 205, 154), 12)
            image_record = cv2.rectangle(image_record, start_point, end_point, (0, 255, 255), 6)
        cv2.imwrite('DSLR_image_with_clicked_Box.jpg', image_record)
        pre_box_center_x = int(box_center_x)
        pre_box_center_y = int(box_center_y)
        app.update()
        
        values[0].delete(0, tk.END)
        values[0].insert(10, x_stage)
        values[1].delete(0, tk.END)
        values[1].insert(10, y_stage)
        values[2].delete(0, tk.END)
        values[2].insert(10, z_reference)
        # call_inject(values)
        panel = tk.Label(app, text='{} Embryo injected'.format(i+1))
        panel.place(x=1600, y=555)
        print(i+1,'Embryo injected')
    LED_on_off(0)


def pip_detection(pipe_left, pipe_right):
    pipe_left_data = pipe_left.queue[-1]
    pipe_right_data = pipe_right.queue[-1]
    if pipe_left_data:
        print(pipe_left_data[0])
        x, y, w, h = pipe_left_data[0]
        pip_left_x = x + (w/2)
        pip_left_y = y + h
        pip_left = np.matrix([[pip_left_x], [pip_left_y]])
    else:
        pip_left = np.matrix([[], []])
    if pipe_right_data:
        print(pipe_right_data[0])
        x, y, w, h = pipe_right_data[0]
        pip_righ_x = x + (w/2)
        pip_righ_y = y + h
        pip_righ = np.matrix([[pip_righ_x], [pip_righ_y]])
    else:
        pip_righ = np.matrix([[], []])
    # pip_left = np.matrix([[543], [194]])
    # pip_righ = np.matrix([[614], [208]])
    print('pip_left', pip_left)
    print('pip_righ', pip_righ)
    return pip_left, pip_righ

def get_z_reference(box_center, dish_number, pip_left, pip_righ, yolk_left, yolk_right):
    box_center_x = box_center[int(len(box_center)/2)][0]
    box_center_y = box_center[int(len(box_center)/2)][1]
    (x_stage, y_stage) = transformation_DSLR_inj(box_center_x, box_center_y, dish_number)
    emb_status = 0
    XYZ = MyXYZ()
    z_reference = 0
    while emb_status == 0:
        z_reference = z_reference + 0.5
        XYZ.Position(x_stage, y_stage, z_reference) # Go to zero
        time.sleep(0.5)
        inj_left, inj_righ, emb_status = embryo_detect(yolk_left, yolk_right, 150, pip_left, pip_righ, 10)
    z_reference = z_reference
    return z_reference

def take_image():
    take_image_flag_1 = [False]
    take_image_flag_2 = [False]

def embryo_detect(yolk_left, yolk_right, pix_range, pip_left, pip_righ, no_yolo_detection):
    all_center_x_left = []
    all_center_y_left = []
    all_center_x_righ = []
    all_center_y_righ = []
    p = 1
    q = 1
    max_p = 1
    max_q = 1
    flag1 = no_yolo_detection
    flag2 = no_yolo_detection
    while (p <= flag1 or q <= flag2) and (max_p <= 50 or max_q <= 50):
        data1 = yolk_left.queue[-1]
        # print('data1', data1)
        data2 = yolk_right.queue[-1]
        # print('data2', data2)
        if data1:
            for j in range(len(data1)):
                x, y, w, h = data1[j]
                center_x = int(x + w/2)
                center_y = int(y + h/2)
                # print('center_x_p = ', center_x)
                if abs(center_x - pip_left.item(0)) <= pix_range:
                    # print('X = ', center_x, 'Y = ', center_y)
                    all_center_x_left.append(center_x)
                    all_center_y_left.append(center_y)
                    # print('p value = ', p)
                    p = p + 1
        max_p = max_p + 1
        if data2:
            for k in range(len(data2)):
                x, y, w, h = data2[k]
                center_x = int(x + w/2)
                center_y = int(y + h/2)
                # print('center_x_q = ', center_x)
                if abs(center_x - pip_righ.item(0)) <= pix_range:
                    # print('X = ', center_x, 'Y = ', center_y)
                    all_center_x_righ.append(center_x)
                    all_center_y_righ.append(center_y)
                    # print('q value = ', q)
                    q = q + 1
        max_q = max_q + 1    
    # If no embryo detected
    if (max_p >= 50 or max_q >= 50):
        inj_left = np.matrix([[], []])
        inj_righ = np.matrix([[], []])
        emb_status = 0
        print('No embryo detected')
        return inj_left, inj_righ, emb_status
    
    avg_center_x_left = np.mean(all_center_x_left)
    avg_center_y_left = np.mean(all_center_y_left)
    avg_center_x_righ = np.mean(all_center_x_righ)
    avg_center_y_righ = np.mean(all_center_y_righ)
    inj_left = np.matrix([[avg_center_x_left], [avg_center_y_left]])
    inj_righ = np.matrix([[avg_center_x_righ], [avg_center_y_righ]])
    emb_status = 1
    return inj_left, inj_righ, emb_status
    
    
def start_whole_camera(values):
    # t = threading.Thread(name='start_whole', target=start_whole)
    # t = threading.Thread(name='start_whole', target=start_onedish_Yolo, args = (q1, q2, values))
    whole_start[0] = True
    yolk_left = Queue()
    yolk_right = Queue()
    pipe_left = Queue()
    pipe_right = Queue()
    t = threading.Thread(name='start_whole', target=start_multidish_Yolo, args = (yolk_left, yolk_right, pipe_left, pipe_right, values))
    w1 = threading.Thread(name='camera_1', target=videoLoop_inclined_1, args = (yolk_left, pipe_left, ))
    w2 = threading.Thread(name='camera_2', target=videoLoop_inclined_2, args = (yolk_right, pipe_right, ))
    # record = threading.Thread(name='Recording', target=recording)
    w1.start()
    w2.start()
    t.start()
    # record.start()
    
def img_DSLR_track(box_coordinate, pre_box_center_x, pre_box_center_y, box_center_x, box_center_y, i, number_detection, emb_status, dish_number):
    image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_with_clicked_Box_{}.jpg'.format(dish_number))
    start_point = (int(box_coordinate[1]), int(box_coordinate[0]))
    end_point = (int(box_coordinate[3]), int(box_coordinate[2]))
    image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 12)
    image = cv2.line(image, (int(pre_box_center_x), int(pre_box_center_y)), (int(box_center_x), int(box_center_y)), (0, 0, 255), 3)
    image_record = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 427))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(image=image)
    panel.image = image
    if dish_number == 1:
        panel.place(x=475, y=25)
    if dish_number == 2:
        panel.place(x=1250, y=25)
    if i == 0:
        image_record = cv2.rectangle(image_record, start_point, end_point, (0, 0, 0), 12)
    elif i == (number_detection - 1):
        image_record = cv2.rectangle(image_record, start_point, end_point, (255, 0, 0), 12)
    elif emb_status == 0:
        image_record = cv2.rectangle(image_record, start_point, end_point, (0, 165, 255), 12)
    else:
        image_record = cv2.rectangle(image_record, start_point, end_point, (50, 205, 154), 12)
        image_record = cv2.rectangle(image_record, start_point, end_point, (0, 255, 255), 6)
    cv2.imwrite('DSLR_image_with_clicked_Box_{}.jpg'.format(dish_number), image_record)
    pre_box_center_x = int(box_center_x)
    pre_box_center_y = int(box_center_y)
    app.update()
    return pre_box_center_x, pre_box_center_y

def start_onedish_Yolo(in_q1, in_q2, values):
    # Start LED
    LED_on_off(3)
    # Take DSLR Image
    DSLR_image('DSLR_image.jpg')
    app.update()
    # Detect embryos
    ML_detect_YOLO('DSLR_image.jpg', 'D:/Microinjection_Project/Python_Code/', 1)
    app.update()
    XYZ = MyXYZ()
    XYZ.Position(-65, 20, 0) # Change this when changing it to two dishe system
    # Sort sequence of embryos Path planning (Add path planning code here)
    box_center = np.load('box_center.npy')
    box_coordinate = np.load('box_coordinate.npy')
    # Initializing variables
    no_yolo_detection = 10
    pix_error_allow = 50
    total_embryos = len(box_center)
    yolo_detection_start[0] = True
    Pressure = MyPressureSystem()
    pre_box_center_x = int(box_center[0][0])
    pre_box_center_y = int(box_center[0][1])
    emb_status = 1
    # For each embryo this for loop works
    for i in range(total_embryos):
        # Read embryo center co ordinate
        box_center_x = box_center[i][0]
        box_center_y = box_center[i][1]
        # Pipette tip detection
        # pip_left = np.matrix([[543], [194]])
        # pip_righ = np.matrix([[614], [208]])
        pip_left, pip_righ = pip_detection(1, 2)
        # Transform to 2 microscope view
        (x_stage, y_stage) = transformation_DSLR_4x(box_center_x, box_center_y)
        curr_inj = np.matrix([[x_stage], [y_stage], [z_reference]])
        XYZ.Position(x_stage, y_stage, z_reference)
        time.sleep(1)
        # Change DSLR image
        pre_box_center_x, pre_box_center_y = img_DSLR_track(box_coordinate[i], pre_box_center_x, pre_box_center_y, box_center_x, box_center_y, i, total_embryos, emb_status)
        #  Detect embryos for the first time
        inj_left, inj_righ, emb_status = embryo_detect(in_q1, in_q2, 350, pip_left, pip_righ, no_yolo_detection)
        pre_box_center_x, pre_box_center_y = img_DSLR_track(box_coordinate[i], pre_box_center_x, pre_box_center_y, box_center_x, box_center_y, i, total_embryos, emb_status)
        # print(inj_left, inj_righ)
        if (inj_left.size > 0 and inj_righ.size > 0):
            attempt = 0
            if abs(inj_left.item(0) - pip_left.item(0)) <= pix_error_allow and abs(inj_righ.item(0) - pip_righ.item(0)) <= pix_error_allow:
                left_curren = np.matrix([[x_stage], [y_stage], [z_reference]])
            while (abs(inj_left.item(0) - pip_left.item(0)) >= pix_error_allow or abs(inj_righ.item(0) - pip_righ.item(0)) >= pix_error_allow) and attempt < 5:
                left_curren, righ_curren = transformation_inj_pip(pip_left, pip_righ, inj_left, inj_righ, curr_inj)
                print(left_curren.item(0), left_curren.item(1), left_curren.item(2))
                XYZ.Position(left_curren.item(0), left_curren.item(1), left_curren.item(2))
                time.sleep(2)
                old_inj_left = inj_left
                old_inj_righ = inj_righ
                inj_left, inj_righ, emb_status = embryo_detect(in_q1, in_q2, 350, pip_left, pip_righ, no_yolo_detection)
                if (inj_left.size == 0 and inj_righ.size == 0):
                    inj_left = old_inj_left
                    inj_righ = old_inj_righ
                    break
                attempt = attempt + 1
                print('Attempt # = ', attempt)

            x_change, y_change, z_change = transformation_pip_z(inj_left, inj_righ, pip_left, pip_righ)
            XYZ.Position(left_curren.item(0) + x_change, left_curren.item(1) + y_change, left_curren.item(2) + z_change)
            time.sleep(1)
            call_pressure(values, 'I')
            time.sleep(1)
            XYZ.Position(left_curren.item(0) + x_change, left_curren.item(1) + y_change, left_curren.item(2) - 3*z_change)
            time.sleep(1)
    XYZ.Position(left_curren.item(0), left_curren.item(1), 0)
    XYZ.Position(0, 0, 0)    
    # Turn off LED
    LED_on_off(0)
    
def start_multidish_Yolo(yolk_left, yolk_right, pipe_left, pipe_right, values):
    # Start LED
    LED_on_off(3)
    injection_material = ['Red dye', 'Blue dye']
    for dish_number in range(1,3):
        # Take DSLR Image
        if dish_number == 1:
            x_place = 475
            y_place = 25
            z_reference = 11.8
        if dish_number == 2:
            x_place = 1250
            y_place = 25
            z_reference = 12.1
        DSLR_image('DSLR_image_{}.jpg'.format(dish_number), dish_number)
        app.update()
        # Detect embryos
        ML_detect_YOLO('DSLR_image_{}.jpg'.format(dish_number), 'D:/Microinjection_Project/Python_Code/', dish_number)
        app.update()
        XYZ = MyXYZ()
        Position = XYZ.Get_Pos()
        # XYZ.set_Velocity(50, 50, 25)
        XYZ.Position(Position['1'], Position['2'], 0) # Go to zero
        XYZ.Position(Position['1'] - 45, Position['2'], 0) # Go to center of dish
        # Sort sequence of embryos Path planning (Add path planning code here)
        box_center = np.load('box_center.npy')
        box_coordinate = np.load('box_coordinate.npy')
        class_ids = np.load('class_ids.npy')
        Path = My_path_planning(box_center, box_coordinate, class_ids)
        box_center, box_coordinate, class_ids, total_path_greedy = Path.Greedy_opt2(box_center, box_coordinate, class_ids)
        # Getting solution
        Alive_num = np.count_nonzero(class_ids == 0)
        Total_vol_required = Alive_num * float((values[16]).get())
        extra_vol = 200
        vial(dish_number, Total_vol_required+extra_vol, 10, [False])
        # Initializing variables
        no_yolo_detection = 10
        pix_error_allow = 50
        total_embryos = len(box_center)
        yolo_detection_start[0] = True
        Pressure = MyPressureSystem()
        pre_box_center_x = int(box_center[0][0])
        pre_box_center_y = int(box_center[0][1])
        emb_status = 1
        unsuccess_inj = 0
        success_inj = 0
        # Pipette tip detection
        XYZ.Position(-7, 60, 0)
        time.sleep(1)
        pip_left = np.matrix([[], []])
        pip_righ = np.matrix([[], []])
        while (pip_left[0].size == 0 or pip_righ[0].size == 0):
            pip_left, pip_righ = pip_detection(pipe_left, pipe_right)
            print('Pip_left', pip_left, 'Pip_righ', pip_righ)
            print('Size left', pip_left[0].size, 'Size right', pip_righ[0].size)
        # pip_left = np.matrix([[100], [100]])
        # pip_righ = np.matrix([[100], [100]])
        print('Pipette left = ', pip_left, 'Pipette right = ', pip_righ)
        XYZ.Position(0, 0, 0)
        XYZ.Position(Position['1'] - 45, Position['2'], 0)
        # # Get z_reference
        # z_reference = get_z_reference(box_center, dish_number, pip_left, pip_righ, yolk_left, yolk_right)
        # print('Z reference = ', z_reference)
        # XYZ.Position(Position['1'] - 45, Position['2'], 0)
        # For each embryo this for loop works
        start_time = time.time()
        for i in range(total_embryos):
            if class_ids[i] == 0:
                # Read embryo center co ordinate
                panel = tk.Label(app, text = 'Injection attempted = {}'.format(i+1))
                panel.place(x=x_place+50, y=y_place+470)
                box_center_x = box_center[i][0]
                box_center_y = box_center[i][1]
                # Transform to 2 microscope view
                (x_stage, y_stage) = transformation_DSLR_inj(box_center_x, box_center_y, dish_number)
                curr_inj = np.matrix([[x_stage], [y_stage], [z_reference]])
                XYZ.Position(x_stage, y_stage, z_reference)
                time.sleep(1)
                # Change DSLR image
                pre_box_center_x, pre_box_center_y = img_DSLR_track(box_coordinate[i], pre_box_center_x, pre_box_center_y, box_center_x, box_center_y, i, total_embryos, emb_status, dish_number)
                #  Detect embryos for the first time
                inj_left, inj_righ, emb_status = embryo_detect(yolk_left, yolk_right, 150, pip_left, pip_righ, no_yolo_detection)
                take_image()
                if emb_status == 0:
                    unsuccess_inj = unsuccess_inj + 1
                pre_box_center_x, pre_box_center_y = img_DSLR_track(box_coordinate[i], pre_box_center_x, pre_box_center_y, box_center_x, box_center_y, i, total_embryos, emb_status, dish_number)
                # print(inj_left, inj_righ)
                if (inj_left.size > 0 and inj_righ.size > 0):
                    attempt = 0
                    if abs(inj_left.item(0) - pip_left.item(0)) <= pix_error_allow and abs(inj_righ.item(0) - pip_righ.item(0)) <= pix_error_allow:
                        left_curren = np.matrix([[x_stage], [y_stage], [z_reference]])
                    while (abs(inj_left.item(0) - pip_left.item(0)) >= pix_error_allow or abs(inj_righ.item(0) - pip_righ.item(0)) >= pix_error_allow) and attempt < 5:
                        left_curren, righ_curren = transformation_inj_pip(pip_left, pip_righ, inj_left, inj_righ, curr_inj)
                        print(left_curren.item(0), left_curren.item(1), left_curren.item(2))
                        XYZ.Position(left_curren.item(0), left_curren.item(1), left_curren.item(2))
                        time.sleep(1)
                        old_inj_left = inj_left
                        old_inj_righ = inj_righ
                        inj_left, inj_righ, emb_status = embryo_detect(yolk_left, yolk_right, 150, pip_left, pip_righ, no_yolo_detection)
                        if (inj_left.size == 0 and inj_righ.size == 0):
                            inj_left = old_inj_left
                            inj_righ = old_inj_righ
                            # unsuccess_inj = unsuccess_inj + 1
                            break
                        attempt = attempt + 1
                        print('Attempt # = ', attempt)                       
                    x_change, y_change, z_change = transformation_pip_z(inj_left, inj_righ, pip_left, pip_righ)
                    # XYZ.set_Velocity(50, 50, 0.5)
                    XYZ.Position(left_curren.item(0) + x_change, left_curren.item(1) + y_change, left_curren.item(2) + z_change + 0.25)
                    time.sleep(0.1)
                    call_pressure(values, 'I')
                    time.sleep(0.1)
                    XYZ.Position(left_curren.item(0) + x_change, left_curren.item(1) + y_change, left_curren.item(2) - 3*z_change)
                    time.sleep(0.1)
                    # XYZ.set_Velocity(50, 50, 25)
                    success_inj = success_inj + 1
                current_time = str(datetime.timedelta(seconds = (time.time() - start_time)))
                panel = tk.Label(app, text='Total injection time = {} sec'.format(current_time))
                panel.place(x=x_place+50, y=y_place+490)
                panel = tk.Label(app, text = 'Unsuccessful Injection = {}'.format(unsuccess_inj))
                panel.place(x=x_place+450, y=y_place+470)
                panel = tk.Label(app, text = 'Successful Injection = {}'.format(success_inj))
                panel.place(x=x_place+250, y=y_place+470)
        time.sleep(1)
        Position = XYZ.Get_Pos()
        XYZ.Position(Position['1'], Position['2'], 0)
        XYZ.Position(0, 0, 0)
        vial(3, 3*extra_vol, 10, [True])
        Data_save(dish_number, injection_material, values, current_time, unsuccess_inj, success_inj)
    # Turn off LED
    LED_on_off(0)
    values[0].delete(0, tk.END)
    values[0].insert(10, 0)
    values[1].delete(0, tk.END)
    values[1].insert(10, 0)
    values[2].delete(0, tk.END)
    values[2].insert(10, 0)
    # Turn of cameras
    camera_off_inclined_1(videoloop_stop_inclined_1)
    camera_off_inclined_2(videoloop_stop_inclined_2)
    # stop_recording()
    # record.join()


fields = 'Position X', 'Position Y', 'Position Z'
fields_d = 'dX', 'dY', 'dZ'
ml_field = ['ML threashold']
fields_UMP = 'Position X', 'Position Y', 'Position Z', 'Position D', 'Speed'
fields_d_UMP = 'dX', 'dY', 'dZ', 'dD'
fields_p_def = 'Vol (nl)', 'Rate (nl/sec)'
ml_threashold = 50
z_reference = 15.5
default = [0, 0, 0]
default_vel = [50, 50, 25]
default_center = [0, 0, 0]
default_DSLR = [102, 11, 13]
default_DSLR_1 = [8, 53, 13]
default_DSLR_2 = [83, -20, 13]
default_d = [0.1, 0.1, 0.1]
default_ml = [50]
default_UMP = [10000, 10000, 10000, 10000, 10000]
default_d_UMP = [100, 100, 100, 100]
default_center_UMP = [10000, 10000, 10000, 10000, 10000]
default_needle = [10000, 10000, 10000, 10000, 10000]
default_needle_XYZ = [-40, 20, 0]
pressure_def = [30, 6]
default_inject = [-76.939, 3.04, 20.35]
videoloop_stop = [False]
autofocus_start = [False]
needle_detection_start_1 = [False]
needle_detection_start_2 = [False]
videoloop_stop_inclined_1 = [False]
videoloop_stop_inclined_2 = [False]
take_image_flag_1 = [False]
take_image_flag_2 = [False]
needle_point_detected = [False]
yolo_detection_start = [False]
whole_start = [False]
inject_status = [True]
recording_stop = [False]

# x_needle = 0
# y_needle = 0
# default_d_10 = [10, 10, 10]
# default_d_50 = [50, 50, 50]
# default_d_100 = [100, 100, 100]
# default_d_500 = [500, 500, 500]
# default_d_1000 = [1000, 1000, 1000]
num = len(fields)
num_d = len(fields_d)
num_UMP = len(fields_UMP)
num_d_UMP = len(fields_d_UMP)
num_p_def = len(fields_p_def)
num_ml = 1
values = []
number = 1

# Getting initial values of XYZ stage and UMP Manipulator
XYZ = MyXYZ()
Position = XYZ.Get_Pos()
Velocity = XYZ.Get_Vel()
default = [round(Position['1'], 2), round(Position['2'], 2), round(Position['3'], 2)]
default_vel = [round(Velocity['1'], 2), round(Velocity['2'], 2), round(Velocity['3'], 2)]
Position = MyUMP.Get_Pos()
default_UMP = [Position['x'], Position['y'], Position['z'], Position['d'], 10000]

for i in range(0, num):
   tk.Label(app, text=fields[i]).grid(row=i+3)
   e = tk.Entry(app)
   e.config(width=7)
   e.insert(0, default[i])
   e.grid(row=i+3, column=1)
   values.append(e)

for j in range(0, num_d):
    tk.Label(app, text=fields_d[j]).grid(row=j+3, column=3)
    z = tk.Entry(app)
    z.config(width=5)
    z.insert(0, default_d[j])
    z.grid(row = j+3, column=4)
    values.append(z)
    
for k in range(0, num_ml):
    tk.Label(app, text=ml_field[k]).grid(row=k+2, column=3)
    y = tk.Entry(app)
    y.config(width=5)
    y.insert(0, default_ml[k])
    y.grid(row = k+2, column=4)
    values.append(y)
    
for l in range(0, num_UMP):
   tk.Label(app, text=fields_UMP[l]).grid(row=l+13)
   e = tk.Entry(app)
   e.config(width=7)
   e.insert(0, default_UMP[l])
   e.grid(row=l+13, column=1)
   values.append(e)

for m in range(0, num_d_UMP):
    tk.Label(app, text=fields_d_UMP[m]).grid(row=m+13, column=3)
    z = tk.Entry(app)
    z.config(width=7)
    z.insert(0, default_d_UMP[m])
    z.grid(row = m+13, column=4)
    values.append(z)
    
for n in range(0, num_p_def):
    tk.Label(app, text=fields_p_def[n]).grid(row=11+n, column=4)
    z = tk.Entry(app)
    z.config(width=5)
    z.insert(0, pressure_def[n])
    z.grid(row = 11+n, column=5)
    values.append(z)
    
for p in range(0, num):
    z = tk.Entry(app)
    z.config(width=5)
    z.insert(0, default_vel[n])
    z.grid(row = p+3, column=2)
    values.append(z)


img_up = ImageTk.PhotoImage(Image.open("GUI_Image\\Up_arrow.png"))
img_down = ImageTk.PhotoImage(Image.open("GUI_Image\Down_arrow.png"))
img_left = ImageTk.PhotoImage(Image.open("GUI_Image\Left_arrow.png"))
img_right = ImageTk.PhotoImage(Image.open("GUI_Image\Right_arrow.png"))
img_center = ImageTk.PhotoImage(Image.open("GUI_Image\Center.png"))

b1 = tk.Button(app, text='DSLR camera1', command=lambda: DSLR_image('DSLR_image_1.jpg', 1))
b1.grid(row=0, column=3, sticky=tk.W, pady=4)
b2 = tk.Button(app, text='ML detect1', command=lambda: ML_detect_YOLO('DSLR_image_1.jpg', 'D:/Microinjection_Project/Python_Code/', 1))
b2.grid(row=0, column=4, sticky=tk.W, pady=4)
b3 = tk.Button(app, text='XYZ Stage', command=lambda: select_XYZ())
b3.grid(row=2, column=1, sticky=tk.W, pady=4)
b4 = tk.Button(app, text='Go to Position', command=lambda: go_to_pos_GUI(values))
b4.grid(row=2, column=0, sticky=tk.W, pady=4)
# b5 = tk.Button(app, text='Camera On', command=lambda: camera_on(videoloop_stop))
# b5.grid(row=0, column=2, sticky=tk.W, pady=4)
b6 = tk.Button(app, text='Center', image = img_center, height = 60, width = 60, command=lambda: move_center(default_center, values))
b6.grid(row=8, column=1, sticky=tk.W, pady=4)
b7 = tk.Button(app, text='X-', image = img_up, height = 60, width = 60, command=lambda: move_x_negative(values))
b7.grid(row=7, column=1, sticky=tk.W, pady=4)
b8 = tk.Button(app, text='X+', image = img_down, height = 60, width = 60, command=lambda: move_x_plus(values))
b8.grid(row=9, column=1, sticky=tk.W, pady=4)
b9 = tk.Button(app, text='Y-', image = img_left, height = 60, width = 60, command=lambda: move_y_negative(values))
b9.grid(row=8, column=0, sticky=tk.W, pady=4)
b10 = tk.Button(app, text='Y+', image = img_right, height = 60, width = 60, command=lambda: move_y_plus(values))
b10.grid(row=8, column=2, sticky=tk.W, pady=4)
b11 = tk.Button(app, text='Z+', image = img_up, height = 60, width = 60, command=lambda: move_z_plus(values))
b11.grid(row=7, column=4, sticky=tk.W, pady=4)
b12 = tk.Button(app, text='Z-', image = img_down, height = 60, width = 60, command=lambda: move_z_negative(values))
b12.grid(row=9, column=4, sticky=tk.W, pady=4)
b13 = tk.Button(app, text='__Z__', height = 4, width = 8, anchor = tk.CENTER)
b13.grid(row=8, column=4, sticky=tk.W, pady=4)
b14 = tk.Button(app, text='0.1', anchor = tk.CENTER, height = 1, width = 4,  command=lambda: change_dx(0.1))
b14.grid(row=2, column=5, sticky=tk.W, pady=4)
b15 = tk.Button(app, text='0.5', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_dx(0.5))
b15.grid(row=3, column=5, sticky=tk.W, pady=4)
b16 = tk.Button(app, text='1', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_dx(1))
b16.grid(row=4, column=5, sticky=tk.W, pady=4)
b17 = tk.Button(app, text='5', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_dx(5))
b17.grid(row=5, column=5, sticky=tk.W, pady=4)
b18 = tk.Button(app, text='LED 1', command=lambda: LED_on_off(1))
b18.grid(row=0, column=0, sticky=tk.W, pady=4)
b19 = tk.Button(app, text='LED 2', command=lambda: LED_on_off(2))
b19.grid(row=0, column=1, sticky=tk.W, pady=4)
b20 = tk.Button(app, text='UMP', command=lambda: select_UMP())
b20.grid(row=18, column=1, sticky=tk.W, pady=4)
b21 = tk.Button(app, text='UMP Calibrate', command=lambda: calibrate_UMP())
b21.grid(row=18, column=2, sticky=tk.W, pady=4)
b22 = tk.Button(app, text='Go to Position', command=lambda: go_to_UMP_pos_GUI(values))
b22.grid(row=18, column=0, sticky=tk.W, pady=4)
b23 = tk.Button(app, text='Center', image = img_center, height = 60, width = 60, command=lambda: move_center_UMP(default_center_UMP, values))
b23.grid(row=20, column=1, sticky=tk.W, pady=4)
b24 = tk.Button(app, text='X+', image = img_up, height = 60, width = 60, command=lambda: move_x_plus_UMP(values))
b24.grid(row=19, column=1, sticky=tk.W, pady=4)
b25 = tk.Button(app, text='X-', image = img_down, height = 60, width = 60, command=lambda: move_x_negative_UMP(values))
b25.grid(row=21, column=1, sticky=tk.W, pady=4)
b26 = tk.Button(app, text='Y-', image = img_left, height = 60, width = 60, command=lambda: move_y_negative_UMP(values))
b26.grid(row=20, column=0, sticky=tk.W, pady=4)
b27 = tk.Button(app, text='Y+', image = img_right, height = 60, width = 60, command=lambda: move_y_plus_UMP(values))
b27.grid(row=20, column=2, sticky=tk.W, pady=4)
b28 = tk.Button(app, text='Z+', image = img_up, height = 60, width = 60, command=lambda: move_z_plus_UMP(values))
b28.grid(row=19, column=3, sticky=tk.W, pady=4)
b29 = tk.Button(app, text='Z-', image = img_down, height = 60, width = 60, command=lambda: move_z_negative_UMP(values))
b29.grid(row=21, column=3, sticky=tk.W, pady=4)
b30 = tk.Button(app, text='__Z__', height = 4, width = 8, anchor = tk.CENTER)
b30.grid(row=20, column=3, sticky=tk.W, pady=4)
b31 = tk.Button(app, text='D+', image = img_up, height = 60, width = 60, command=lambda: move_d_plus_UMP(values))
b31.grid(row=19, column=4, sticky=tk.W, pady=4)
b32 = tk.Button(app, text='D-', image = img_down, height = 60, width = 60, command=lambda: move_d_negative_UMP(values))
b32.grid(row=21, column=4, sticky=tk.W, pady=4)
b33 = tk.Button(app, text='__D__', height = 4, width = 8, anchor = tk.CENTER)
b33.grid(row=20, column=4, sticky=tk.W, pady=4)
b34 = tk.Button(app, text='0.1', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_dx_UMP(0.1))
b34.grid(row=13, column=5, sticky=tk.W, pady=4)
b35 = tk.Button(app, text='10', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_dx_UMP(10))
b35.grid(row=14, column=5, sticky=tk.W, pady=4)
b36 = tk.Button(app, text='100', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_dx_UMP(100))
b36.grid(row=15, column=5, sticky=tk.W, pady=4)
b37 = tk.Button(app, text='1000', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_dx_UMP(1000))
b37.grid(row=16, column=5, sticky=tk.W, pady=4)
b38 = tk.Button(app, text='100', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_speed_UMP(100))
b38.grid(row=17, column=3, sticky=tk.W, pady=4)
b39 = tk.Button(app, text='1000', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_speed_UMP(1000))
b39.grid(row=17, column=4, sticky=tk.W, pady=4)
b40 = tk.Button(app, text='10000', anchor = tk.CENTER, height = 1, width = 4, command=lambda: change_speed_UMP(10000))
b40.grid(row=17, column=5, sticky=tk.W, pady=4)
# b41 = tk.Button(app, text='Camera off', command=lambda: camera_off(videoloop_stop), bg='red')
# b41.grid(row=1, column=2, sticky=tk.W, pady=4)
# b42 = tk.Button(app, text='Auto focus', command=lambda: call_autofocus(values))
# b42.grid(row=10, column=0, sticky=tk.W, pady=4)
b43 = tk.Button(app, text='Camera On 1', command=lambda: camera_on_inclined_1(videoloop_stop_inclined_1))
b43.grid(row=11, column=0, sticky=tk.W, pady=4)
b44 = tk.Button(app, text='Camera Off 1', command=lambda: camera_off_inclined_1(videoloop_stop_inclined_1), bg='red')
b44.grid(row=11, column=1, sticky=tk.W, pady=4)
# b45 = tk.Button(app, text='Needle Position', command=lambda: needle_position())
# b45.grid(row=10, column=2, sticky=tk.W, pady=4)
# b46 = tk.Button(app, text='Needle Detection 2', command=lambda: call_needle_detection_2(values))
# b46.grid(row=12, column=2, sticky=tk.W, pady=4)
b47 = tk.Button(app, text='Inject', command=lambda: inject_fun(values))
b47.grid(row=10, column=3, sticky=tk.W, pady=4)
b48 = tk.Button(app, text='Pressure', command=lambda: get_inject_status(True), bg='green')
b48.grid(row=11, column=3, sticky=tk.W, pady=4)
b49 = tk.Button(app, text='Withdraw', command=lambda: get_inject_status(False), bg='red')
b49.grid(row=12, column=3, sticky=tk.W, pady=4)
b50 = tk.Button(app, text='Start', command=lambda: start_whole_camera(values), bg='green')
b50.grid(row=10, column=1, sticky=tk.W, pady=4)
b51 = tk.Button(app, text='Camera On 2', command=lambda: camera_on_inclined_2(videoloop_stop_inclined_2))
b51.grid(row=12, column=0, sticky=tk.W, pady=4)
b52 = tk.Button(app, text='Camera Off 2', command=lambda: camera_off_inclined_2(videoloop_stop_inclined_2), bg='red')
b52.grid(row=12, column=1, sticky=tk.W, pady=4)
# b53 = tk.Button(app, text='Needle Detection 1', command=lambda: call_needle_detection_1(values))
# b53.grid(row=11, column=2, sticky=tk.W, pady=4)
b54 = tk.Button(app, text='Yolo', command=lambda: call_ml_yolo(values))
b54.grid(row=10, column=4, sticky=tk.W, pady=4)
b55 = tk.Button(app, text='2 LEDs on', command=lambda: LED_on_off(3))
b55.grid(row=1, column=0, sticky=tk.W, pady=4)
b56 = tk.Button(app, text='2 LEDs off', command=lambda: LED_on_off(0))
b56.grid(row=1, column=1, sticky=tk.W, pady=4)
b57 = tk.Button(app, text='DSLR camera2', command=lambda: DSLR_image('DSLR_image_2.jpg', 2))
b57.grid(row=1, column=3, sticky=tk.W, pady=4)
b58 = tk.Button(app, text='ML detect2', command=lambda: ML_detect_YOLO('DSLR_image_2.jpg', 'D:/Microinjection_Project/Python_Code/', 2))
b58.grid(row=1, column=4, sticky=tk.W, pady=4)
b59 = tk.Button(app, text='Vial 1', command=lambda: vial(1, float((values[16]).get()), float((values[17]).get()), inject_status))
b59.grid(row=6, column=0, sticky=tk.W, pady=4)
b60 = tk.Button(app, text='Vial 2', command=lambda: vial(2, float((values[16]).get()), float((values[17]).get()), inject_status))
b60.grid(row=6, column=1, sticky=tk.W, pady=4)
b61 = tk.Button(app, text='Vial 3', command=lambda: vial(3, float((values[16]).get()), float((values[17]).get()), inject_status))
b61.grid(row=6, column=2, sticky=tk.W, pady=4)
b62 = tk.Button(app, text='Vial 4', command=lambda: vial(4, float((values[16]).get()), float((values[17]).get()), inject_status))
b62.grid(row=6, column=3, sticky=tk.W, pady=4)
b63 = tk.Button(app, text='Set speed', command=lambda: XYZ_set_speed(values))
b63.grid(row=2, column=2, sticky=tk.W, pady=4)



# b1 = Button(app, text='Quit', command=app.quit)
# b1.grid(row=6, column=0, sticky=W, pady=4)
# b2 = Button(app, text='Go to Position', command=lambda: go_to_pos_GUI(values))
# b2.grid(row=6, column=1, sticky=W, pady=4)
# b3 = Button(app, text='Camera On', command=lambda: camera_on_off(1))
# b3.grid(row=7, column=0, sticky=W, pady=4)
# #b4 = Button(app, text='UMP On', command=lambda: UMP_Call(speed, depth, X, Y))
# #b4.grid(row=7, column=1, sticky=W, pady=4)
# b5 = Button(app, text='Auto Focus', command=lambda: focus_XYZ(values))
# b5.grid(row=8, column=0, sticky=W, pady=4)
# b6 = Button(app, text='UP', image = img_up, command=lambda: move2x_up(values))
# b6.grid(row=6, column=3, sticky=W, pady=4)
# b7 = Button(app, text='DOWN', image = img_down, command=lambda: move2x_down(values))
# b7.grid(row=8, column=3, sticky=W, pady=4)
# b8 = Button(app, text='LEFT', image = img_left, command=lambda: move2x_left(values))
# b8.grid(row=7, column=2, sticky=W, pady=4)
# b9 = Button(app, text='RIGHT', image = img_right, command=lambda: move2x_right(values))
# b9.grid(row=7, column=4, sticky=W, pady=4)
# b10 = Button(app, text='_10_', anchor = CENTER, command=lambda: change_2x4x(1, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
# b10.grid(row=0, column=5, sticky=W, pady=4)
# b11 = Button(app, text='_50_', anchor = CENTER, command=lambda: change_2x4x(2, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
# b11.grid(row=1, column=5, sticky=W, pady=4)
# b12 = Button(app, text='_100', anchor = CENTER, command=lambda: change_2x4x(3, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
# b12.grid(row=2, column=5, sticky=W, pady=4)
# b13 = Button(app, text='_500', anchor = CENTER, command=lambda: change_2x4x(4, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
# b13.grid(row=3, column=5, sticky=W, pady=4)
# b14 = Button(app, text='1000', anchor = CENTER, command=lambda: change_2x4x(5, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
# b14.grid(row=4, column=5, sticky=W, pady=4)
# b15 = Button(app, text='Z_up', image = img_up, command=lambda: move2x_z_up(values))
# b15.grid(row=6, column=5, sticky=W, pady=4)
# b16 = Button(app, text='Z_down', image = img_down, command=lambda: move2x_z_down(values))
# b16.grid(row=8, column=5, sticky=W, pady=4)
# b17 = Button(app, text='__Z__', anchor = CENTER)
# b17.grid(row=7, column=5, sticky=W, pady=4)
# b18 = Button(app, text='Center', image = img_center, command=lambda: move_center(default_center, values))
# b18.grid(row=7, column=3, sticky=W, pady=4)
# b19 = Button(app, text='Cali', command=lambda: Calibration())
# b19.grid(row=9, column=5, sticky=W, pady=4)

target=app.bind('<Button 3>', getorigin)
app.mainloop()