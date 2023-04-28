'''
import serial
import time

arduino = serial.Serial('COM3',9600)
time.sleep(2)

print arduino.readline()
print ("Enter '1' to turn on the LED and '0' to turn LED off")

while 1:

    var = 1
    #var = raw_inpur()
    print("You Entered :", var)

    if(var == '1'):
        arduino.write('1')
        print("LED turned on")
        time.sleep(1)

    if(var == '0'):
        arduino.write('0')
        print("LED turned off")
'''

import serial
import time
import numpy as np

arduino = serial.Serial('COM6', 9600, timeout = 5)                                         #Creates arduino object and establishes connection to port (Enter your port)
time.sleep(2)                                                                               #waits for connection to establish

print arduino.readline()
print ("Enter '1' to turn 'on' the LED and '0' to turn LED 'off'")      #asks for input

while 1:                                                                                        #while data is available
    var = raw_input()                                                                         #accepts input puts it in variable 'var'
    #var = 1
    print "You Entered :", var                                                             #prints input
    '''
    arduino.write('1')
    print("LED turned on")
    time.sleep(2)
    arduino.write('0')
    print("LED turned off")
    time.sleep(2)
    '''

    if(var == '1'):                                                                                #if input is 1
        arduino.write('1')                                                                          #sends '1' to arduino
        print("LED turned on")
        time.sleep(1)
    if(var == '0'):                                                                                 #if var is 0
        arduino.write('0')                                                                          #sends '0' to arduino
        print("LED turned off")