# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:48:04 2019

@author: enet-joshi317-admin
"""

import sympy as sym
import numpy as np

def function_transformation_matrix_DSLR_4x(embryo_x,embryo_y,px1,py1,px2,py2,px3,py3,X1,Y1,X2,Y2,X3,Y3):
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
#embryo_coords=function_transformation_matrix_DSLR_4x(61,21,-69,11,-43,-21,60,23,-41,23,-49,28,58,-35)
#
#needle_coords=function_transformation_matrix_DSLR_4x(415,440,355,487,396,490,430,463,10296,27960,10296,28130,10296,28140)
#embryo_coords=function_transformation_matrix_DSLR_4x(2300,320,2032,2234,2110,2203,2141,2279,67363,84048,66753,82618,68153,82038)
#print('X embryo 4x',int(float(embryo_coords.item(0,0))))
#print('Y embryo 4x',int(float(embryo_coords.item(1,0))))
##embryo_coords=function_transformation_matrix_DSLR_4x(0,0,3831,1588,1981,1641,3428,3272,50043,69858,52543,108068,85043,76810)
#embryo_coords = function_transformation_matrix_DSLR_4x(1849,1155,1860,2817,3041,1433,1038,1204,66833,92088,37453,66298,32103,109048)
#print('X embryo 4x',int(float(embryo_coords.item(0,0))))
#print('Y embryo 4x',int(float(embryo_coords.item(1,0))))