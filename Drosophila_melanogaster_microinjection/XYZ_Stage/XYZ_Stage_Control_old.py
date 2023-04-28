# Allmotion Python Serial port example
# Verified on Win 10  With CP2101 USB UART Bridge Running On COM5
# Verified on Anaconda Spyder 2.7

# may need to do do pip install pyserial at command prompt

import serial
import time


# ser.close()

def serial_ports():
    ports = ['COM%s' % (i + 1) for i in range(256)]

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


# See which com port shows up and change "COM5" in code below to that com port
if __name__ == '__main__':
    print(serial_ports())
    COM_connected = serial_ports()
    print(COM_connected)

    ser = serial.Serial(
        port='COM8',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
Vx = 1000
Vy = 1000
Vz = 1000
X = 1000
Y = 1000
Z = 0
#str = '/1V%d,%d,%daM1A%d,%d,%dR\r\n'%(Vx,Vy,Vz,X,Y,Z)
str = '/1V%d,%d,%dD%d,%d,%dR\r\n'%(Vx,Vy,Vz,X,Y,Z)
print(str)
#ser.write('/1P1000D1000R\r\n')
ser.write(str)
time.sleep(5)
'''
ser.write('/1V500P1000V500D1000R\r\n')
time.sleep(5)
ser.write('/1V2000,2000P1000,1000V2000,2000D1000,1000R\r\n')
time.sleep(2.5)
ser.write('/1V1000,1000,10000P1000,1000,10000V1000,1000,10000D1000,1000,10000R\r\n')
time.sleep(2.5)
'''
ser.write('/1aM1A5000,5000,5000R\r\n')
time.sleep(5)
#ser.write('/1aM1A4000R\r\n')
#ser.write('/1aM1A000,000,0000R\r\n')
#ser.write('/1aM1R\r\n')
#xyz = ser.write('/1?aM1AR\r\n')
#print(xyz)

ser.close()