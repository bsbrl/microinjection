# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:37:17 2020

@author: enet-joshi317-admin
"""

import numpy
def Delta_XY_FOV_2_lin(needle_x,embryo_x):
    Delta_embryo_needle_xs=[0,-107,104]
    Delta_Xs=[0,200,-200]
    coeffs=numpy.polyfit(Delta_embryo_needle_xs,Delta_Xs,1)
    Delta_X=-(coeffs[0]*(embryo_x-needle_x)+coeffs[1])
    Delta_Y=(float(32)/float(100))*Delta_X
    return Delta_X,Delta_Y
# dx,dy=Delta_XY_FOV_2_lin(1,2)