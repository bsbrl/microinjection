import serial
import time

def injection(PSI, wait_time):
    arduino = serial.Serial('COM7', 9600, timeout = 5)
    value = int(PSI)
    time.sleep(5)
    if wait_time=='bp':
        print "Sending", PSI, "back pressure through needle"
        signal = (value + 0.35)/0.107
        signal = str(signal)
        P = "P"
        p = "p"
        signal = P + signal + p
        print (signal)
        arduino.write(signal)       
    else:        
        if value <=5:
            signal = (value + 0.35)/0.107
            signal = str(signal)
            P = "P"
            p = "p"
            signal = P + signal + p
            print (signal)
            arduino.write(signal)
            time.sleep(wait_time)
            arduino.write("P0p")
            print "Done with", PSI, "PSI pressure."
    
        if value > 5:
            signal = (value + 43.6279)/0.9535
            signal = str(signal)
            P = "P"
            p = "p"
            signal = P + signal + p
            print (signal)
            arduino.write(signal)
            time.sleep(wait_time)
            arduino.write("P0p")
            print "Done with", PSI, "PSI pressure."

#injection(45,5)
#injection(45,10)