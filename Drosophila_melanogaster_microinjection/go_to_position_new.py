# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:46:24 2022

@author: me-alegr011-admin
"""

from XYZ_Stage.XYZ_Position import XYZ_Location
import serial
# import os
# import time
import numpy as np

def go_to_position_new():
    # time.sleep(5)
    # os.system("taskkill /python.exe")
    try:
      ser = serial.Serial(
        port="COM5",
        baudrate=9600,
      )
      ser.isOpen() # try to open port, if possible print message and proceed with 'while True:'
      print ("port is opened!")
    
    except IOError: # if port is already opened, close it and open it again and print message
      ser.close()
      ser.open()
      print ("port was already open, was closed and opened again!")
    vals=np.load('currentXY.npy')
    XYZ_Location(11250,11250,8000,int(vals[0]),int(vals[1]),0,ser)
    ser.close()