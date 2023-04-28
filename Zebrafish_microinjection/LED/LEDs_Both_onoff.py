import serial
import time
import numpy as np

def LED(value):
    arduino = serial.Serial('COM7', 9600, timeout = 5) #open the arduino code, open USB port
    time.sleep(1.75) #just repeat the line
    L = "L" 
    l = "l"
    if value == 0:
        value = str(value)
        signal = str.encode(L+value+l)
        arduino.write(signal)
        print("Turning off both LEDs")
    if value == 1:
        value = str(value)
        signal = str.encode(L+value+l) #make the signal, for two LED, I need two sets of signals
        #print(signal)
        arduino.write(signal) #send the signal from USB port to arduino
        print("Turning on LED1, and turning off LED2")
        #time.sleep(0.1)       
    if value == 2:
        value = str(value)
        signal = str.encode(L+value+l)
        #print(signal)
        arduino.write(signal)
        print("Turning on LED2, and turning off LED1")
        #time.sleep(0.1)
    if value == 3:
        value = str(value)
        signal = str.encode(L+value+l)
        arduino.write(signal)
        print("Turning on both LEDs")
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
    return


# LED(0)

