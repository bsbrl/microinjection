# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 14:59:58 2021

@author: enet-joshi317-admin
"""
import numpy as np
def rowcol_fun(x1,x2,y1,y2):
    end=0
    next_col=0
    c=1
    while end==0:
        if np.mean([x1,x2])<=400+next_col:
            end=1
        else:
            next_col+=400
            c+=1
    end=0
    next_row=0
    r=1
    while end==0:
        if np.mean([y1,y2])<=400+next_row:
            end=1
        else:
            next_row+=400
            r+=1
    return r,c
#r,c=rowcol_fun(1370,1430,1973,2028)