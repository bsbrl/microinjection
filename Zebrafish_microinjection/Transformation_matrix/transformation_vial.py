# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:50:45 2021

@author: admin
"""


import numpy as np

def transformation_vial(vial_num):
    if vial_num == 1:
        x = -22
        y = -4.6
        z = 17
    if vial_num == 2:
        x = -13.2
        y = 4.5
        z = 17
        x = -13.5
        y = 5.4
        z = 17
    if vial_num == 3:
        x = 5.5
        y = 22.9
        z = 17
    if vial_num == 4:
        x = 14.8
        y = 32
        z = 17
        
    return x, y, z