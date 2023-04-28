# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:17:17 2020

@author: admin
"""


import cv2

def camera_inclined(number):
    if number == 1 or number == 0 or number == 2:
        cap = cv2.VideoCapture(number)
        if not(cap.isOpened()):
            print('Could not open video device')
        else:
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
            cap.set(cv2.CAP_PROP_FPS,30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS,1)
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                # Display the resulting frame
                cv2.imshow('preview',frame)
                #Waits for a user input to quit the application
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.imwrite('image_captured.jpg',frame)
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                
    if number == 'both':
        cap1 = cv2.VideoCapture(1)
        cap2 = cv2.VideoCapture(2)
        if not(cap1.isOpened()):
            print('Could not open video device 1')
        elif not(cap1.isOpened()):
            print('Could not open video device 2')
        else:
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*2)
            cap1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
            cap1.set(cv2.CAP_PROP_FPS,30)
            cap1.set(cv2.CAP_PROP_AUTOFOCUS,1)
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
            cap2.set(cv2.CAP_PROP_FPS,30)
            cap2.set(cv2.CAP_PROP_AUTOFOCUS,1)
            while(True):
                # Capture frame-by-frame
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                #frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
                #frame2 = cv2.rotate(frame2, cv2.ROTATE_180)
                # Display the resulting frame
                cv2.imshow('preview1',frame1)
                cv2.imshow('preview2',frame2)
                #Waits for a user input to quit the application
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.imwrite('image_captured1.jpg',frame1)
                    cap1.release()
                    cv2.imwrite('image_captured2.jpg',frame2)
                    cap2.release()
                    cv2.destroyAllWindows()
                    break
            
camera_inclined('both')