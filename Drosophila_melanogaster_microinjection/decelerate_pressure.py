# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 17:26:14 2021

@author: User
"""

from Pressure_Control.Continuous_Pressure import continuous_pressure

def decelerate_pressure(arduino,pressure_value,back_pressure_value):
    press_count=0
    back_pressure_value_new=pressure_value
    while back_pressure_value_new!=back_pressure_value:
        back_pressure_value_new=pressure_value-(press_count)
        print(back_pressure_value_new)
        l='x'
        l_o='x'
        l_e='1'
        k=0
        while l=='x' or l_o=='x' or l_e!='p':
            print('Try ',k+1)
            signal=continuous_pressure(back_pressure_value_new,pressure_value,'bp')
            arduino.write(signal.encode())
            l_=arduino.readline()
            l_=l_.decode()
            print(l_)
            l_o=l_[7]
            l=l_[8]
            l_e=l_[len(l_)-3]
            k+=1
        press_count+=1
# import serial
# import time
# arduino = serial.Serial('COM7', 9600, timeout = 5)
# time.sleep(5)
# signal_1=continuous_pressure(30,30,'inj')
# arduino.write(signal_1.encode())
# time.sleep(5)
# decelerate_pressure(arduino,30,15)
# time.sleep(5)
# arduino.write("P0p".encode())