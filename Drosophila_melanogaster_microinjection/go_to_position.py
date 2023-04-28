# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:38:55 2022

@author: User
"""
from XYZ_Stage.XYZ_Position import XYZ_Location

def go_to_position(values,ser):
   XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()),
                int((values[4]).get()), int((values[5]).get()),ser)
