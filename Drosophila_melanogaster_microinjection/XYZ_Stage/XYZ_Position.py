import serial
import time

def XYZ_Calibration():
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
    
    print(serial_ports())

    ser = serial.Serial\
    (
        port='COM5',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
    str = '/1s0aM1f1m34h17aM2f1m34h17aM3f1m50h25L571,571,8100V888,888,2520aM2n0Z16000aM3n0Z50000aM1n0Z12000e1R'
    ser.write(str.encode())
    time.sleep(3)
    str = '/1s1V100,100,200aM2Z16000n2aM3Z50000n2aM1Z12000n2L57,57,810V32000,32000,48000A800,1070,2520M5000R'
    ser.write(str.enocde())
    time.sleep(2)
    ser.close()
    
def XYZ_Location(Vx,Vy,Vz,X,Y,Z,ser):
    
    Vx = int((Vx * 88.89)/1000)
    Vy = int((Vy * 88.89)/1000)
    Vz = int((Vz * 1260)/1000)
    X = int((X * 88.89)/1000)
    Y = int((Y * 88.89)/1000)
    Z = int((Z * 1260)/1000)

    str = '/1V%d,%d,%daM1A%d,%d,%dR\r\n'%(Vx,Vy,Vz,X,Y,Z)
    ser.write(str.encode())
    if X == 0 & Y == 0 & Z == 0:
        sleep_time = 5
    else:
        sleep_time = max(X/Vx,Y/Vy,Z/Vz)+0.5
    #time.sleep(sleep_time)

def XYZ_Positive(Vx,Vy,Vz,X,Y,Z):
    
    Vx = (Vx * 88.89)/1000
    Vy = (Vy * 88.89)/1000
    Vz = (Vz * 1260)/1000
    X = (X * 88.89)/1000
    Y = (Y * 88.89)/1000
    Z = (Z * 88.89)/1000
    
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
    print(serial_ports())

    ser = serial.Serial\
    (
        port='COM5',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
    print(Vx,Vy,Vz,X,Y,Z)
    if X == 0 or Y == 0 or Z == 0:
        if X == 0 and Y == 0 and Z == 0:
            str = '0'
        elif X == 0 and Y == 0:
            Vx = 0
            Vy = 0
            str = '/1V,,%dP,,%dR\r\n' % (Vz, Z)
        elif X == 0 and Z == 0:
            Vx = 0
            Vz = 0
            str = '/1V,%dP,%dR\r\n' % (Vy, Y)
        elif Y == 0 and Z == 0:
            Vy = 0
            Vz = 0
            str = '/1V%dP%dR\r\n' % (Vx, X)
        elif X == 0:
            Vx = 0
            str = '/1V,%d,%dP,%d,%dR\r\n' % (Vy, Vz, Y, Z)
        elif Y == 0:
            Vy = 0
            str = '/1V%d,,%dP%d,,%dR\r\n' % (Vx, Vz, X, Z)
        elif Z == 0:
            Vz = 0
            str = '/1V%d,%dP%d,%dR\r\n' % (Vx, Vy, X, Y)
    else:
        str = '/1V%d,%d,%dP%d,%d,%dR\r\n'%(Vx,Vy,Vz,X,Y,Z)
    print(str)
    ser.write(str.encode())
    sleep_time = max(X/(Vx-1),Y/(Vy-1),Z/(Vz-1))+0.5
    #time.sleep(sleep_time)
    #time.sleep(2)
    ser.close()

def XYZ_Negative(Vx,Vy,Vz,X,Y,Z):
    
    Vx = (Vx * 88.89)/1000
    Vy = (Vy * 88.89)/1000
    Vz = (Vz * 1260)/1000
    X = (X * 88.89)/1000
    Y = (Y * 88.89)/1000
    Z = (Z * 88.89)/1000
    
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
    print(serial_ports())

    ser = serial.Serial\
    (
        port='COM5',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )
    print(Vx, Vy, Vz, X, Y, Z)
    if X == 0 or Y == 0 or Z == 0:
        if X == 0 and Y == 0 and Z == 0:
            str = '0'
        elif X == 0 and Y == 0:
            Vx = 0
            Vy = 0
            str = '/1V,,%dD,,%dR\r\n' % (Vz, Z)
        elif X == 0 and Z == 0:
            Vx = 0
            Vz = 0
            str = '/1V,%dD,%dR\r\n' % (Vy, Y)
        elif Y == 0 and Z == 0:
            Vy = 0
            Vz = 0
            str = '/1V%dD%dR\r\n' % (Vx, X)
        elif X == 0:
            Vx = 0
            str = '/1V,%d,%dD,%d,%dR\r\n' % (Vy, Vz, Y, Z)
        elif Y == 0:
            Vy = 0
            str = '/1V%d,,%dD%d,,%dR\r\n' % (Vx, Vz, X, Z)
        elif Z == 0:
            Vz = 0
            str = '/1V%d,%dD%d,%dR\r\n' % (Vx, Vy, X, Y)
    else:
        str = '/1V%d,%d,%dD%d,%d,%dR\r\n' % (Vx, Vy, Vz, X, Y, Z)
    print(str)
    ser.write(str.encode())
    sleep_time = max(X / (Vx - 1), Y / (Vy - 1), Z / (Vz - 1)) + 0.5
    #time.sleep(sleep_time)
    #time.sleep(2)
    ser.close()