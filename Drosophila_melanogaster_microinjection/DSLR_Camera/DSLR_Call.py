# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:53:09 2019

@author: enet-joshi317-admin
"""

import sys
import os
import subprocess
import datetime 

def func_TakeNikonPicture(input_filename):
    camera_command = 'C:/Program Files (x86)/digiCamControl/CameraControlCmd.exe'
    #amera_command_details = '/filename ./' + input_filename + ' /capture /iso 500 /shutter 1/30 /aperture 1.8'
#    camera_command_details = '/filename ./' + 'DSLR_Camera/' + input_filename + ' /capture /iso 100 /shutter 1/125 /aperture 8'
    camera_command_details = '/filename ./' + 'DSLR_Camera/' + input_filename + ' /capturenoaf /iso 400 /shutter 1/200 /aperture 7.1'
    print('camera details = ',camera_command_details)
    full_command=camera_command + ' ' + camera_command_details
    p = subprocess.Popen(full_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
#    (output, err) = p.communicate()  
##    print(output)
#    #This makes the wait possible
##    p_status = p.wait(1)
##    print(p.stdout.readline())
#
#    #This will give you the output of the command being executed
#    print('Command output: ' + str(output))
#    print('Command err: ' + str(err))
#
#    print('done')
    
#if(len(sys.argv) < 2):
#    rawimagename = 'test.jpg'
#else:   
#    # sys.argv[0] is the program name, sys.argv[1] is the first file, etc.
#    # need to shift this over
#    files = sys.argv[1:len(sys.argv)]
#    # Read the image
#    rawimagename = files[0]
#    if(os.path.isfile(rawimagename) is True):
#        print("File exists...not overwriting.")
#        sys.exit()
#
## Store date/time for file uniqueness
#current_dt=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#print("Current date time = " + current_dt)
#rawimagename=current_dt + '_' + rawimagename

#rawimagename = 'DSLR_Entire_Petri_Dish_3.jpg'
#print('Name of raw image will be: ', rawimagename)

# take picture
# func_TakeNikonPicture('Entire_Petri_Dish_testing.jpg')
# import cv2
# import numpy as np
# image=cv2.imread('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/DSLR_Camera/Entire_Petri_Dish_testing.jpg')
# image_height=6000
# image_width=4000
# center=(int((float(image_width))/(2)),int((float(image_height))/(2)))
# scale=1
# fromCenter=False
# M_1 = cv2.getRotationMatrix2D(center,270, scale)
# cosine = np.abs(M_1[0, 0])
# sine = np.abs(M_1[0, 1])
# nW = int((image_height * sine) + (image_height * cosine))
# nH = int((image_height * cosine) + (image_width * sine))
# M_1[0, 2] += (nW / 2) - int((float(image_width))/(2))
# M_1[1, 2] += (nH / 2) - int((float(image_height))/(2))
# new_1=cv2.warpAffine(image, M_1, (image_height, image_width)) 
# cv2.imwrite('C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/DSLR_Camera/Entire_Petri_Dish_testing_new.jpg',new_1)