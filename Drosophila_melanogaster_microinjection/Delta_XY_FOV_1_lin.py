# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:27:00 2020

@author: enet-joshi317-admin
"""

import numpy
def Delta_XY_FOV_1_lin(needle_x,embryo_x):
    # Delta_embryo_needle_xs=[0,-68,159]
    # Delta_Xs=[0,120,-300]
    Delta_embryo_needle_xs=[0,-205,136]
    Delta_Xs=[0,240,-160]
    coeffs=numpy.polyfit(Delta_embryo_needle_xs,Delta_Xs,1)
    Delta_X=-(coeffs[0]*(embryo_x-needle_x)+coeffs[1])
    Delta_Y=-(float(100)/float(80))*Delta_X
    return Delta_X,Delta_Y
# dx,dy=Delta_XY_FOV_1_lin(1,2)