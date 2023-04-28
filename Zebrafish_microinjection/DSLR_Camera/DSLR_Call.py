# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:53:09 2019

@author: enet-joshi317-admin
"""

import sys
import os
import subprocess
import datetime 

def func_TakeNikonPicture(input_filename, input_path):
    camera_command = 'C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe'
    camera_command_details = '/filename ./' + input_filename + ' /capturenoaf /iso 500 /shutter 1/30 /aperture 1.8'
    # camera_command_details = 'D:/Microinjection_Project/Python_Code ./' + input_filename + ' /capture /iso 500 /shutter 1/30 /aperture 1.8'
    # camera_command_details = '/filename ./' + 'Image/' + input_filename + ' /capture /iso 100 /shutter 1/30 /aperture 12'
    print('camera details = ',camera_command_details)
    full_command=camera_command + ' ' + camera_command_details
    p = subprocess.Popen(full_command, stdout=subprocess.PIPE, universal_newlines=True, shell=False)
    (output, err) = p.communicate()  

    #This makes the wait possible
    #p_status = p.wait(1)
    # print(p.stdout.readline())

    #This will give you the output of the command being executed
    print('Command output: ' + str(output))
    print('Command err: ' + str(err))

    print('done')
    
    # path = input_path
    # os.chdir(path)
    # if os.path.exists(input_filename):
    #     os.remove(input_filename)
    # os.rename('D:/Microinjection_Project/Python_Code/DSLR_Camera/' + input_filename, input_path  + input_filename)
    
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
# func_TakeNikonPicture('DSLR_image_2.jpg', 'D:/Microinjection_Project/Python_Code/')
