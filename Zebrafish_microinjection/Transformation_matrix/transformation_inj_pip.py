# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:16:21 2021

@author: admin
"""


import numpy as np

def transformation_inj_pip(pip_left, pip_righ, inj_left, inj_righ, curr_inj):
    # pip_left = np.matrix([[630], [289]])
    # pip_righ = np.matrix([[512], [306]])
    # inj_left = np.matrix([[869], [375]])
    # inj_righ = np.matrix([[741], [714]])
    
    # curr_inj = np.matrix([[-66], [21.15], [10.5]])
    
    # left_x = np.matrix([[-3] , [-3]])
    # left_y = np.matrix([[-2.63], [2.63]])
    # righ_x = np.matrix([[2.76], [-2.76]])
    # righ_y = np.matrix([[-2.26], [-2.26]])
    
    # left_x = np.matrix([[-2.21], [-3.39]])
    # left_y = np.matrix([[-2.80], [2.09]])
    # righ_x = np.matrix([[2.55], [-3.28]])
    # righ_y = np.matrix([[-3.06], [-2.45]])
    
    left_x = np.matrix([[-2.39], [-3.29]])
    left_y = np.matrix([[-2.81], [2.14]])
    righ_x = np.matrix([[2.47], [-3.29]])
    righ_y = np.matrix([[-3.03], [-2.38]])
    
    img_size = np.matrix([[1280], [720]])
    
    all_XYZ = []
    
    for i in range(400):
        pip_left_curr = pip_left + np.matrix([[0], [i]])
        xyz_left = pip_left_curr - inj_left
        left = np.divide(xyz_left, img_size)
        left_change_1 = np.multiply(left[0], left_x)
        left_change_2 = np.multiply(left[1], left_y)
        left_change = left_change_1 + left_change_2
        left_change = np.append(left_change, np.matrix([0]), axis=0)
        left_curren = curr_inj + left_change
        
        pip_righ_curr = pip_righ + np.matrix([[0], [i]])
        xyz_righ = pip_righ_curr - inj_righ
        righ = np.divide(xyz_righ, img_size)
        righ_change_1 = np.multiply(righ[0], righ_x)
        righ_change_2 = np.multiply(righ[1], righ_y)
        righ_change = righ_change_1 + righ_change_2
        righ_change = np.append(righ_change, np.matrix([0]), axis=0)
        righ_curren = curr_inj + righ_change
        
        diff_XYZ = np.absolute(left_curren - righ_curren)
        all_XYZ.append(np.linalg.norm(diff_XYZ))
        
    min_index = np.argmin(all_XYZ)
    
    pip_left_curr = pip_left + np.matrix([[0], [min_index]])
    xyz_left = pip_left_curr - inj_left
    left = np.divide(xyz_left, img_size)
    left_change_1 = np.multiply(left[0], left_x)
    left_change_2 = np.multiply(left[1], left_y)
    left_change = left_change_1 + left_change_2
    left_change = np.append(left_change, np.matrix([0]), axis=0)
    left_curren = curr_inj + left_change
    
    pip_righ_curr = pip_righ + np.matrix([[0], [min_index]])
    xyz_righ = pip_righ_curr - inj_righ
    righ = np.divide(xyz_righ, img_size)
    righ_change_1 = np.multiply(righ[0], righ_x)
    righ_change_2 = np.multiply(righ[1], righ_y)
    righ_change = righ_change_1 + righ_change_2
    righ_change = np.append(righ_change, np.matrix([0]), axis=0)
    righ_curren = curr_inj + righ_change
    return left_curren, righ_curren

def transformation_pip_z(inj_left, inj_righ, pip_left, pip_righ):
    pix_diff = inj_left.item(1) - pip_left.item(1)
    # z_change = (pix_diff/248)
    x_change = (pix_diff/24000)
    y_change = (pix_diff/24000)
    z_change = (pix_diff/242.42)
    return x_change, y_change, z_change

# pip_left = np.matrix([[630], [289]])
# pip_righ = np.matrix([[512], [306]])
# inj_left = np.matrix([[730], [426]])
# inj_righ = np.matrix([[498], [528]])
# curr_inj = np.matrix([[-67], [8.44], [10.5]])
    
# pip_left = np.matrix([[628], [310]])
# pip_righ = np.matrix([[776], [342]])
# inj_left = np.matrix([[461], [658]])
# inj_righ = np.matrix([[451], [455]])
# curr_inj = np.matrix([[-42.4227], [60.34], [13]])
# left_curren, righ_curren = transformation_inj_pip(pip_left, pip_righ, inj_left, inj_righ, curr_inj)
# print(left_curren, righ_curren)