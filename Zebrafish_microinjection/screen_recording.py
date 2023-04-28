# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 15:06:53 2021

@author: admin
"""


from threading import Thread
import time
import cv2
import pyautogui
import numpy as np

# display screen resolution, get it from your OS settings
SCREEN_SIZE = (1920, 1080)
# define the codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
# create the video write object
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (SCREEN_SIZE))
start_time = time.time()

def start_record(mirror = False):
    while True:
        # make a screenshot
        img = pyautogui.screenshot()
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        # convert colors from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # write the frame
        out.write(frame)
        # show the frame
        cv2.imshow("screenshot", frame)
        # if the user clicks q, it exits        
        if record_stop[0] == True:
            cv2.destroyAllWindows()
            out.release()
            break

def stop_record(flag):
    print('stop record')
    record_stop[0] = True

record_stop = [False]
record_thread = Thread(target=start_record)
record_thread.start()
time.sleep(5)
stop_record(1)
# make sure everything is closed when exited
# t = threading.Thread(name='start_whole', target=start_multidish_Yolo, args = (yolk_left, yolk_right, pipe_left, pipe_right, values))
# w1 = threading.Thread(name='camera_1', target=videoLoop_inclined_1, args = (yolk_left, pipe_left, ))
# w2 = threading.Thread(name='camera_2', target=videoLoop_inclined_2, args = (yolk_right, pipe_right, ))
