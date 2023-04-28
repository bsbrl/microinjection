# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:51:11 2020

@author: enet-joshi317-admin
"""

import numpy
def Delta_Z_FOV_2_lin(needle_y,embryo_y):
    # Delta_embryo_needle_ys=[0,-81,-138]
    # Delta_Zs=[0,240,390]
    Delta_embryo_needle_ys=[0,-63,-126]
    Delta_Zs=[0,250,500]
    coeffs=numpy.polyfit(Delta_embryo_needle_ys,Delta_Zs,1)
    Delta_Z=-(coeffs[0]*(embryo_y-needle_y)+coeffs[1])
    return Delta_Z
# dz=Delta_Z_FOV_2_lin(239,286)