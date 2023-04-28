# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:45:25 2020

@author: enet-joshi317-admin
"""

import numpy
def Delta_Z_FOV_1_lin(needle_y,embryo_y):
    # Delta_embryo_needle_ys=[0,-90,-182]
    # Delta_Zs=[0,250,500]
    Delta_embryo_needle_ys=[0,-64,-123]
    Delta_Zs=[0,250,500]
    coeffs=numpy.polyfit(Delta_embryo_needle_ys,Delta_Zs,1)
    Delta_Z=-(coeffs[0]*(embryo_y-needle_y)+coeffs[1])
    return Delta_Z
# dz=Delta_Z_FOV_1_lin(239,286)