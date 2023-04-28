# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:35:10 2020

@author: enet-joshi317-admin
"""

import serial
import time

def back_pressure(PSI):
    arduino = serial.Serial('COM7', 9600, timeout = 5)
    value = int(PSI)
    time.sleep(5)
    print "Sending", PSI, "back pressure through needle"
    signal = (value + 0.35)/0.107
    signal = str(signal)
    P = "P"
    p = "p"
    signal = P + signal + p
    print (signal)
    arduino.write(signal)       
#back_pressure(10)