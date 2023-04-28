# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:39:50 2020

@author: admin
"""

import os
os.chdir('D:/Microinjection_Project/Python_Code/Sensapex_Manipulator')
from sensapex import *
import time
import pandas as pd

class MyUMP():
    def __init__(self):
        flag = 1 
        self.ump = UMP()
        # self.x = x
        # self.y = y
        # self.z = z
        # self.d = d
        # self.speed = speed
        # self.Position()
        #self.ump.close()
        
    def Position(x, y, z, d, speed):
        ump = UMP()
        ump.get_soft_start_state(1)
        pos = ump.get_pos(1, timeout=0)
        print('Moving with speed = ',speed,'towards\nx =',x,', y =',y,', z =',z,', d =',d)
        pos[0] = x*1000
        pos[1] = y*1000
        pos[2] = z*1000
        pos[3] = d*1000
        ump.set_max_acceleration(1, 0)
        ump.goto_pos(1, pos, speed=speed, simultaneous=True, linear=False, max_acceleration=0)
        while ump.is_busy(1) == True:
            time.sleep(0.1)
        print('Movement finished\n')
        ump.close()
    
    def Calibration(yes_no):
        if yes_no == True:
            ump = UMP()
            print('Calibration started')
            ump.calibrate_zero_position(1)
            time.sleep(70)
            print('Calibration finished\n')
            ump.close()
    
    def Get_Pos():
        ump = UMP()
        pos = ump.get_pos(1, timeout = 0)
        x = pos[0]/1000
        y = pos[1]/1000
        z = pos[2]/1000
        d = pos[3]/1000
        Position = pd.DataFrame([x, y, z, d], index=['x', 'y', 'z', 'd'])[0]
        # Position = [x, y, z, d]
        ump.close()
        return Position
        
        
# ump_go(10000, 10000, 10000, 10000, 3000)
# ump_go(1000, 1000, 1000, 1000, 3000)
# MyUMP.Position(9400, 10000, 8800, 10000, 10000)
# Position = MyUMP.Get_Pos()
# print(Position)