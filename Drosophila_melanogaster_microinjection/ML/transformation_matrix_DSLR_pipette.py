# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:07:56 2021

@author: User
"""

import sympy as sym
import numpy as np

def function_transformation_matrix_DSLR_pipette(embryo_x,embryo_y,px1,py1,px2,py2,px3,py3,X1,Y1,X2,Y2,X3,Y3):
    # x,y pixels
    point_xy_1=[px1,py1]
    point_xy_2=[px2,py2]
    point_xy_3=[px3,py3]
    
    # y,z stage coordinates
    point_xy_stage_1=[X1,Y1]
    point_xy_stage_2=[X2,Y2]
    point_xy_stage_3=[X3,Y3]
    
    a11_xy,a12_xy,a21_xy,a22_xy,c1_xy,c2_xy = sym.symbols('a11_xy,a12_xy,a21_xy,a22_xy,c1_xy,c2_xy')
    
    eqn1_xy=sym.Eq(c1_xy+a11_xy*point_xy_1[0]+a12_xy*point_xy_1[1],point_xy_stage_1[0])
    eqn2_xy=sym.Eq(c2_xy+a21_xy*point_xy_1[0]+a22_xy*point_xy_1[1],point_xy_stage_1[1])
    eqn3_xy=sym.Eq(c1_xy+a11_xy*point_xy_2[0]+a12_xy*point_xy_2[1],point_xy_stage_2[0])
    eqn4_xy=sym.Eq(c2_xy+a21_xy*point_xy_2[0]+a22_xy*point_xy_2[1],point_xy_stage_2[1])
    eqn5_xy=sym.Eq(c1_xy+a11_xy*point_xy_3[0]+a12_xy*point_xy_3[1],point_xy_stage_3[0])
    eqn6_xy=sym.Eq(c2_xy+a21_xy*point_xy_3[0]+a22_xy*point_xy_3[1],point_xy_stage_3[1])
    
    xy=sym.solve([eqn1_xy,eqn2_xy,eqn3_xy,eqn4_xy,eqn5_xy,eqn6_xy],(a11_xy,a12_xy,a21_xy,a22_xy,c1_xy,c2_xy))
    
#    print('Matrix = ',np.matrix([[ (float(xy[a11_xy])), (float(xy[a12_xy]))],
#                    [ (float(xy[a21_xy])),  (float(xy[a22_xy]))]]))
    embryo_coords= np.array([[float(xy[c1_xy])],[float(xy[c2_xy])]])+np.matrix([[ float(xy[a11_xy]), float(xy[a12_xy])],
                    [ float(xy[a21_xy]),  float(xy[a22_xy])]])*np.array([[float(embryo_x)],[float(embryo_y)]])
    
    return embryo_coords
# embryo_coords = function_transformation_matrix_DSLR_pipette(2040,2602,2979,1769,978,1447,1591,2288,23180,18750,65580,25050,52420,7500)
# print('X embryo 4x',int(float(embryo_coords.item(0,0))))
# print('Y embryo 4y',int(float(embryo_coords.item(1,0))))