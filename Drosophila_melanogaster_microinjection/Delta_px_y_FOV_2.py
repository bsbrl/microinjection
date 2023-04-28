# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:36:59 2022

@author: User
"""

import numpy

def Delta_px_y_FOV_2(dz):
    Delta_embryo_needle_ys=[0,-63,-126]
    Delta_Zs=[0,250,500]
    coeffs=numpy.polyfit(Delta_Zs,Delta_embryo_needle_ys,1)
    dy=-(coeffs[0]*(dz)+coeffs[1])
    return dy
# dy=Delta_px_y_FOV_2(53.68371802869633)