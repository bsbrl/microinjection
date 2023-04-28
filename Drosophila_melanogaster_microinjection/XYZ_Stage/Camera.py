import numpy as np
import cv2

def image(number):
    cap = cv2.VideoCapture(1) # video capture source camera (Here webcam of laptop)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while(True):
        ret,frame = cap.read()
        cv2.imshow('img1',frame) #display the captured image
        #if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
        cv2.imwrite('C:/Users/asjos/Anaconda2/Projects/XYZ Stage/Images/starting_pos_needle_trial_{}.jpg'.format(number),frame)
        cv2.destroyAllWindows()
        break

    cap.release()