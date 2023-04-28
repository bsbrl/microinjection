import serial
import time
import numpy as np

def LED(value):
    arduino = serial.Serial('COM6', 9600, timeout = 5)
    time.sleep(1.75)
    L = "L"
    l = "l"
    if value == 1:
        value = str(value)
        signal = str.encode(L+value+l)
        #print(signal)
        arduino.write(signal)
        print("Turning on LED")
        #time.sleep(0.1)
        
    if value == 0:
        value = str(value)
        signal = str.encode(L+value+l)
        #print(signal)
        arduino.write(signal)
        print("Turning off LED")
        #time.sleep(0.1)
    '''    
    print "Turning on LED for ", wait_time, "sec"
    value = int(wait_time*1000)
    value = str(value)
    L = "L"
    l = "l"
    signal = L+value+l
    print(signal)
    print "Switching on LED for ", wait_time, "sec"
    arduino.write(signal)
    time.sleep(wait_time)
    print "Switching off LED after ", wait_time, "sec"
    '''

# LED(0)