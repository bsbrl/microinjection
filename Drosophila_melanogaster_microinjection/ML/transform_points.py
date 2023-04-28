# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:27:14 2022

@author: User
"""

def transform_points(resize_width,resize_height,actual_width,actual_height,x_resize,y_resize):
    x=int(((x_resize)*actual_width)/resize_width)
    y=int(((y_resize)*actual_height)/resize_height)
    return x,y