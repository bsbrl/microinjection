# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:17:10 2020

@author: admin
"""


from __future__ import print_function

from pipython import GCSDevice, pitools
import time

class MyXYZ():
    def __init__(self):
        # Defining controller name, stage, and reference mode
        self.CONTROLLERNAME = 'C-884'  # 'C-884' will also work
        self.STAGES = ['L-731.093132_X', 'L-731.093132_Y', 'L-310.023232', 'NOSTAGE']
        self.REFMODES = ['FNL', 'FRF']
        
        # Now connecting to controller through USB port
        with GCSDevice(self.CONTROLLERNAME) as pidevice:
            # Change serialnum value according to USB connection address
            pidevice.ConnectUSB(serialnum='120035601') 
            # Printing connected values
            # print('connected: {}'.format(pidevice.qIDN().strip()))
            # Print version information 
            # if pidevice.HasqVER():
            #     print('version info:\n{}'.format(pidevice.qVER().strip()))
            # Initializing stages connected to controller 
            # print('initialize connected stages...')
            pitools.startup(pidevice, stages=self.STAGES, refmodes=self.REFMODES)
            # Defining max min and current position of XYZ stage
            rangemin = pidevice.qTMN()
            rangemax = pidevice.qTMX()
            curpos = pidevice.qPOS()
            # print('Done initializing stages')
            
    def MoveZero(self):
        print('Moving stage to (0, 0, 13) location')
        with GCSDevice(self.CONTROLLERNAME) as pidevice:
            pidevice.ConnectUSB(serialnum='120035601') 
            pitools.startup(pidevice, stages=self.STAGES, refmodes=self.REFMODES)
            axis = pidevice.axes
            pidevice.MOV(axis[0], 0)
            pidevice.MOV(axis[1], 0)
            pidevice.MOV(axis[2], 13)
            pitools.waitontarget(pidevice, axes=axis)
            position = pidevice.qPOS(axis)
            print('Current position of axis {} is {}'.format(axis[0], position["1"]))
            print('Current position of axis {} is {}'.format(axis[1], position["2"]))
            print('Current position of axis {} is {}'.format(axis[2], position["3"]))
            
    def Position(self, X, Y, Z):
        if X >= -102.5 and X <= 102.5 and Y >= -102.5 and Y <= 102.5 and Z >= 0 and Z <= 26:
            # print('Moving stage to ({}, {}, {}) location'.format(X,Y,Z))
            with GCSDevice(self.CONTROLLERNAME) as pidevice:
                pidevice.ConnectUSB(serialnum='120035601') 
                pitools.startup(pidevice, stages=self.STAGES, refmodes=self.REFMODES)
                axis = pidevice.axes
                pidevice.MOV(axis[0], X)
                pidevice.MOV(axis[1], Y)
                pidevice.MOV(axis[2], Z)
                pitools.waitontarget(pidevice, axes=axis)
                position = pidevice.qPOS(axis)
                # print('Current position of all axes are ({}, {}, {})'.format(position["1"], position["2"], position["3"]))
        else:
            print('X, Y, Z values are out of range')
    
    def set_Velocity(self, X_vel, Y_vel, Z_vel):
        if X_vel >= 0 and X_vel <= 100 and Y_vel >= 0 and Y_vel <= 100 and Z_vel >= 0 and Z_vel <= 50:
            with GCSDevice(self.CONTROLLERNAME) as pidevice:
                pidevice.ConnectUSB(serialnum='120035601') 
                pitools.startup(pidevice, stages=self.STAGES, refmodes=self.REFMODES)
                axis = pidevice.axes
                pidevice.VEL(axis[0], X_vel)
                pidevice.VEL(axis[1], Y_vel)
                pidevice.VEL(axis[2], Z_vel)
            print('Velocity of XYZ stage set to = (',X_vel, Y_vel, Z_vel, ')')
        else:
            print('Velocity of X, Y, Z values are out of range')
            
    def Get_Pos(self):
        with GCSDevice(self.CONTROLLERNAME) as pidevice:
            pidevice.ConnectUSB(serialnum='120035601') 
            pitools.startup(pidevice, stages=self.STAGES, refmodes=self.REFMODES)
            axis = pidevice.axes
            position = pidevice.qPOS(axis)
        return (position)
    
    def Get_Vel(self):
        with GCSDevice(self.CONTROLLERNAME) as pidevice:
            pidevice.ConnectUSB(serialnum='120035601') 
            pitools.startup(pidevice, stages=self.STAGES, refmodes=self.REFMODES)
            axis = pidevice.axes
            velocity = pidevice.qVEL(axis)
        return (velocity)
        
            
# XYZ = MyXYZ()
# XYZ.set_Velocity(50, 50, 25)
# XYZ.Position(0,0,0)
# Position = XYZ.Get_Pos()
# print(Position)
# XYZ.set_Velocity(10, 10, 10)
# XYZ.Position(100,50,0)
# XYZ.Position(0,0,0.005)
# XYZ.Position(0,0,0.01)
# XYZ.Position(0,0,0.015)
        
# XYZ = MyXYZ()
# XYZ.Position(0,0,0)
# velocity = XYZ.Get_Vel()
# print(velocity)
# XYZ.set_Velocity(10, 10, 10)
# velocity = XYZ.Get_Vel()
# print(velocity)