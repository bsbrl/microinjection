# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:54:41 2021

@author: User
"""

import sympy as sym
import numpy as np

def function_transformation_matrix_embryo_guess(dX,dY,dZ,dpx1_1,dpy1_1,dpx1_2,dpy1_2,dpx2_1,dpy2_1,dpx2_2,dpy2_2,dpx3_1,dpy3_1,dpx3_2,dpy3_2,dpx4_1,dpy4_1,dpx4_2,dpy4_2,dX1,dY1,dZ1,dX2,dY2,dZ2,dX3,dY3,dZ3,dX4,dY4,dZ4):
    # Need 4 points
    # x,y pixels
    point_xy_1=[dpx1_1,dpy1_1,dpx1_2,dpy1_2]
    point_xy_2=[dpx2_1,dpy2_1,dpx2_2,dpy2_2]
    point_xy_3=[dpx3_1,dpy3_1,dpx3_2,dpy3_2]
    point_xy_4=[dpx4_1,dpy4_1,dpx4_2,dpy4_2]
    
    # y,z stage coordinates
    point_xy_stage_1=[dX1,dY1,dZ1]
    point_xy_stage_2=[dX2,dY2,dZ2]
    point_xy_stage_3=[dX3,dY3,dZ3]
    point_xy_stage_4=[dX4,dY4,dZ4]
    
    a11_xy,a12_xy,a13_xy,a21_xy,a22_xy,a23_xy,a31_xy,a32_xy,a33_xy,a41_xy,a42_xy,a43_xy,c1_xy,c2_xy,c3_xy,c4_xy = sym.symbols('a11_xy,a12_xy,a13_xy,a21_xy,a22_xy,a23_xy,a31_xy,a32_xy,a33_xy,a41_xy,a42_xy,a43_xy,c1_xy,c2_xy,c3_xy,c4_xy')
    
    eqn1_xy=sym.Eq(c1_xy+a11_xy*point_xy_stage_1[0]+a12_xy*point_xy_stage_1[1]+a13_xy*point_xy_stage_1[2],point_xy_1[0])
    eqn2_xy=sym.Eq(c2_xy+a21_xy*point_xy_stage_1[0]+a22_xy*point_xy_stage_1[1]+a23_xy*point_xy_stage_1[2],point_xy_1[1])
    eqn3_xy=sym.Eq(c3_xy+a31_xy*point_xy_stage_1[0]+a32_xy*point_xy_stage_1[1]+a33_xy*point_xy_stage_1[2],point_xy_1[2])
    eqn4_xy=sym.Eq(c4_xy+a41_xy*point_xy_stage_1[0]+a42_xy*point_xy_stage_1[1]+a43_xy*point_xy_stage_1[2],point_xy_1[3])
    
    eqn5_xy=sym.Eq(c1_xy+a11_xy*point_xy_stage_2[0]+a12_xy*point_xy_stage_2[1]+a13_xy*point_xy_stage_2[2],point_xy_2[0])
    eqn6_xy=sym.Eq(c2_xy+a21_xy*point_xy_stage_2[0]+a22_xy*point_xy_stage_2[1]+a23_xy*point_xy_stage_2[2],point_xy_2[1])
    eqn7_xy=sym.Eq(c3_xy+a31_xy*point_xy_stage_2[0]+a32_xy*point_xy_stage_2[1]+a33_xy*point_xy_stage_2[2],point_xy_2[2])
    eqn8_xy=sym.Eq(c4_xy+a41_xy*point_xy_stage_2[0]+a42_xy*point_xy_stage_2[1]+a43_xy*point_xy_stage_2[2],point_xy_2[3])
    
    eqn9_xy=sym.Eq(c1_xy+a11_xy*point_xy_stage_3[0]+a12_xy*point_xy_stage_3[1]+a13_xy*point_xy_stage_3[2],point_xy_3[0])
    eqn10_xy=sym.Eq(c2_xy+a21_xy*point_xy_stage_3[0]+a22_xy*point_xy_stage_3[1]+a23_xy*point_xy_stage_3[2],point_xy_3[1])
    eqn11_xy=sym.Eq(c3_xy+a31_xy*point_xy_stage_3[0]+a32_xy*point_xy_stage_3[1]+a33_xy*point_xy_stage_3[2],point_xy_3[2])
    eqn12_xy=sym.Eq(c4_xy+a41_xy*point_xy_stage_3[0]+a42_xy*point_xy_stage_3[1]+a43_xy*point_xy_stage_3[2],point_xy_3[3]) 
    
    eqn13_xy=sym.Eq(c1_xy+a11_xy*point_xy_stage_4[0]+a12_xy*point_xy_stage_4[1]+a13_xy*point_xy_stage_4[2],point_xy_4[0])
    eqn14_xy=sym.Eq(c2_xy+a21_xy*point_xy_stage_4[0]+a22_xy*point_xy_stage_4[1]+a23_xy*point_xy_stage_4[2],point_xy_4[1])
    eqn15_xy=sym.Eq(c3_xy+a31_xy*point_xy_stage_4[0]+a32_xy*point_xy_stage_4[1]+a33_xy*point_xy_stage_4[2],point_xy_4[2])
    eqn16_xy=sym.Eq(c4_xy+a41_xy*point_xy_stage_4[0]+a42_xy*point_xy_stage_4[1]+a43_xy*point_xy_stage_4[2],point_xy_4[3])
    
    xy=sym.solve([eqn1_xy,eqn2_xy,eqn3_xy,eqn4_xy,eqn5_xy,eqn6_xy,eqn7_xy,eqn8_xy,eqn9_xy,eqn10_xy,eqn11_xy,eqn12_xy,eqn13_xy,eqn14_xy,eqn15_xy,eqn16_xy],(a11_xy,a12_xy,a13_xy,a21_xy,a22_xy,a23_xy,a31_xy,a32_xy,a33_xy,a41_xy,a42_xy,a43_xy,c1_xy,c2_xy,c3_xy,c4_xy))
    embryo_coords= np.array([[float(xy[c1_xy])],[float(xy[c2_xy])],[float(xy[c3_xy])],[float(xy[c4_xy])]])+np.matrix([[ float(xy[a11_xy]), float(xy[a12_xy]),float(xy[a13_xy])],
                    [ float(xy[a21_xy]),  float(xy[a22_xy]),float(xy[a23_xy])],[ float(xy[a31_xy]),  float(xy[a32_xy]),float(xy[a33_xy])],[ float(xy[a41_xy]),  float(xy[a42_xy]),float(xy[a43_xy])]])*np.array([[float(dX)],[float(dY)],[float(dZ)]])
    
    return embryo_coords

# dX=-30
# dY=-20
# dZ=-10
# # px1_1=1
# # py1_1=2
# # px1_2=3
# # py1_2=4
# # px2_1=5
# # py2_1=2
# # px2_2=6
# # py2_2=2
# # px3_1=3
# # py3_1=2
# # px3_2=8
# # py3_2=5
# # px4_1=2
# # py4_1=6
# # px4_2=4
# # py4_2=2
# # X1=4
# # Y1=1
# # Z1=2
# # X2=38
# # Y2=5
# # Z2=2
# # X3=2
# # Y3=5
# # Z3=8
# # X4=6
# # Y4=21
# # Z4=2
# x_fov_1_orig=560
# y_fov_1_orig=323
# x_fov_2_orig=748
# y_fov_2_orig=330
# # x_fov_1_orig=0
# # y_fov_1_orig=0
# # x_fov_2_orig=0
# # y_fov_2_orig=0
# # embryo_coords = function_transformation_matrix_embryo_guess(dX,dY,dZ,-37, -44, -25, -8, -26, -35, -41, 3, 10, -16, 1, -21, -22, -31, -14, -8,25,-10,30,40,20,-25,-30,40,-10,15,-20,10)
# embryo_coords = function_transformation_matrix_embryo_guess(dX,dY,dZ,0,0,0,0, -26, -35, -41, 3, 10, -16, 1, -21, -22, -31, -14, -8,0,0,0,40,20,-25,-30,40,-10,15,-20,10)
# print('x_fov_1 = ',x_fov_1_orig+int(float(embryo_coords.item(0,0))))
# print('y_fov_1 = ',y_fov_1_orig+int(float(embryo_coords.item(1,0))))
# print('x_fov_2 = ',x_fov_2_orig+int(float(embryo_coords.item(2,0))))
# print('y_fov_2 = ',y_fov_2_orig+int(float(embryo_coords.item(3,0))))