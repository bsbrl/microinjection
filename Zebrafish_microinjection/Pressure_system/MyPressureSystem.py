# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:43:27 2020

@author: admin
"""


import serial
import time
import numpy as np

class MyPressureSystem():
    def __init__(self):
        flag = 1
        self.ser = serial.Serial()
        
    def set_target(self,volume):
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = 'COM5'
        ser.open()
        volume = str(volume)
        signal = str.encode('V'+volume+'\r\n')
        ser.write(signal)
        line = ser.readline()
        print(line)
        ser.close()
        
    def set_rate(self,rate):
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = 'COM5'
        ser.open()
        rate = str(rate)
        signal = str.encode('R'+rate+'\r\n')
        ser.write(signal)
        line = ser.readline()
        print(line)
        ser.close()
        
    def set_direction(self,direction):
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = 'COM5'
        ser.open()
        signal = str.encode(direction + '\r\n')
        ser.write(signal)
        line = ser.readline()
        print(line)
        ser.close()
        
    def inject(self,volume,rate):
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = 'COM5'
        print('Injecting', volume, 'nl at a rate of', rate, 'nl/sec')
        ser.open()
        time_left = float(volume/rate)
        vol_str = str(volume)
        rat_str = str(rate)
        signal = str.encode('V' + vol_str + '\r\n')
        ser.write(signal)
        signal = str.encode('R' + rat_str + '\r\n')
        ser.write(signal)
        signal = str.encode('I' + '\r\n')
        ser.write(signal)
        signal = str.encode('G' + '\r\n')
        ser.write(signal)
        time.sleep(time_left)
        print('Injection of liquid done')
        ser.close()
        
    def withdraw(self,volume,rate):
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.port = 'COM5'
        print('Withdrawing', volume, 'nl at a rate of', rate, 'nl/sec')
        ser.open()
        time_left = float(volume/rate)
        vol_str = str(volume)
        rat_str = str(rate)
        signal = str.encode('V' + vol_str + '\r\n')
        ser.write(signal)
        signal = str.encode('R' + rat_str + '\r\n')
        ser.write(signal)
        signal = str.encode('W' + '\r\n')
        ser.write(signal)
        signal = str.encode('G' + '\r\n')
        ser.write(signal)
        time.sleep(time_left)
        print('Withdrawal of liquid done')
        ser.close()

# Pressure(2)
# Pressure = MyPressureSystem()
# Pressure.set_target(60)
# Pressure.set_rate(60)
# Pressure.set_direction('W')
# Pressure.inject(1000,1000)
# Pressure.withdraw(100,10)