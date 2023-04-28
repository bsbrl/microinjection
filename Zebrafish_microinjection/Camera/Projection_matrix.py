# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:39:32 2022

@author: asjos
"""

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from numpy.linalg import matrix_rank
import cv2
import matplotlib.pyplot as plt
import random

df = pd.read_csv('Calibration_data_collection_ProjectionMatrix_all.csv')
data = df.values
all_data_dt = pd.DataFrame(data).to_numpy()

# all_data_dt = np.load('Calibration_data_collection_ProjectionMatrix.npy')

# Creating A matrix. 
'''
Ax = 0
A = |0           0           0       -dX        -dY       -1      dy1 dX     dy1 dY     dy1|
    |dX          dY          1        0          0         0     -dx1 dX    -dx1 dY    -dx1|
    |-dy1 dX   -dy1 dY     -dy1      dx1 dX     dx1 dY     dx1       0          0         0|
    |0           0           0       -dX        -dY       -1      dy2 dX     dy2 dY     dy2|
    |dX          dY          1        0          0         0     -dx2 dX    -dx2 dY    -dx2|
    |-dy2 dX   -dy2 dY     -dy2      dx2 dX     dx2 dY     dx2       0          0         0|
'''

data = np.copy(all_data_dt) 

A = np.empty([1, 9])
for i in range(len(all_data_dt)):
    current_A1 = np.array([0, 0, 0, -data[i,4], -data[i,5], -1, data[i,1]*data[i,4], data[i,1]*data[i,5], data[i,1]])
    current_A2 = np.array([data[i,4], data[i,5], 1, 0, 0, 0, -data[i,0]*data[i,4], -data[i,0]*data[i,5], -data[i,0]])
    current_A3 = np.array([-data[i,1]*data[i,4], -data[i,1]*data[i,5], -data[i,1], data[i,0]*data[i,4], data[i,0]*data[i,5], data[i,0], 0, 0, 0])
    current_A4 = np.array([0, 0, 0, -data[i,4], -data[i,5], -1, data[i,3]*data[i,4], data[i,3]*data[i,5], data[i,3]])
    current_A5 = np.array([data[i,4], data[i,5], 1, 0, 0, 0, -data[i,2]*data[i,4], -data[i,2]*data[i,5], -data[i,2]])
    current_A6 = np.array([-data[i,3]*data[i,4], -data[i,3]*data[i,5], -data[i,3], data[i,2]*data[i,4], data[i,2]*data[i,5], data[i,2], 0, 0, 0])
    current_A = np.vstack([current_A1, current_A2, current_A3, current_A4, current_A5, current_A6])
    A = np.vstack([A, current_A])
    
A = np.delete(A, (0), axis=0)
A = np.asmatrix(A)
P, D, Q = np.linalg.svd(A, full_matrices=False)
A_try = P @ np.diag(D) @ Q
Proj_mat = np.array([[Q[8,0], Q[8,1], Q[8,2]], [Q[8,3], Q[8,4], Q[8,5]], [Q[8,6], Q[8,7], Q[8,8]]])
Proj_mat_inv = np.linalg.inv(Proj_mat)

# Running RANSACtp get better Projection matrix
total_iter = 10000
inlines = np.zeros((total_iter, 1))
all_Proj_matrix = []
iter_numb =  0
threshold = 0.6
rand_numb = len(all_data_dt)
while (iter_numb < total_iter):
    if iter_numb % 100 == 0:
        print('Current iter number', iter_numb)
    flag = 0
    A_mat = np.empty([1, 9])
    list_rand = random.sample(range(0, rand_numb), 20)
    for i in list_rand:
        current_A1 = np.array([0, 0, 0, -data[i,4], -data[i,5], -1, data[i,1]*data[i,4], data[i,1]*data[i,5], data[i,1]])
        current_A2 = np.array([data[i,4], data[i,5], 1, 0, 0, 0, -data[i,0]*data[i,4], -data[i,0]*data[i,5], -data[i,0]])
        current_A3 = np.array([-data[i,1]*data[i,4], -data[i,1]*data[i,5], -data[i,1], data[i,0]*data[i,4], data[i,0]*data[i,5], data[i,0], 0, 0, 0])
        current_A4 = np.array([0, 0, 0, -data[i,4], -data[i,5], -1, data[i,3]*data[i,4], data[i,3]*data[i,5], data[i,3]])
        current_A5 = np.array([data[i,4], data[i,5], 1, 0, 0, 0, -data[i,2]*data[i,4], -data[i,2]*data[i,5], -data[i,2]])
        current_A6 = np.array([-data[i,3]*data[i,4], -data[i,3]*data[i,5], -data[i,3], data[i,2]*data[i,4], data[i,2]*data[i,5], data[i,2], 0, 0, 0])
        current_A = np.vstack([current_A1, current_A2, current_A3, current_A4, current_A5, current_A6])
        A_mat = np.vstack([A_mat, current_A])
    A_mat = np.delete(A_mat, (0), axis=0)
    A_mat = np.asmatrix(A_mat)
    P_mat, D_mat, Q_mat = np.linalg.svd(A_mat, full_matrices=True)
    Proj_mat = np.array([[Q_mat[8,0], Q_mat[8,1], Q_mat[8,2]], [Q_mat[8,3], Q_mat[8,4], Q_mat[8,5]], [Q_mat[8,6], Q_mat[8,7], Q_mat[8,8]]])
    # Check number of inliners
    for j in range(len(all_data_dt)):
        dx1 = all_data_dt[j, 0]
        dy1 = all_data_dt[j, 1]
        dx2 = all_data_dt[j, 2]
        dy2 = all_data_dt[j, 3]
        dX = all_data_dt[j, 4]
        dY = all_data_dt[j, 5]
        Pix_diff = np.array([[0, -1, dy1], [1, 0, -dx1], [-dy1, dx1, 0], [0, -1, dy2], [1, 0, -dx2], [-dy2, dx2, 0]])
        XY_diff = np.array([[dX], [dY], [1]])
        err = Pix_diff @ Proj_mat @ XY_diff
        if np.linalg.norm(err) < threshold:
            flag = flag + 1
    inlines[iter_numb] = flag
    all_Proj_matrix.append(Proj_mat)
    
    iter_numb = iter_numb + 1

idx = np.argmax(inlines)
print('Number of highest inlines', inlines[idx])
final_proj_matrix = all_Proj_matrix[idx]



'''
total_iter = 100000;
inlines = np.zeros((total_iter,1));
F_matrix = [];
iter_numb = 0;
threshold = 0.01;
rand_numb = len(pts1)
    while (iter_numb < total_iter):
        if iter_numb % 1000 == 0:
            print('Current iter number', iter_numb)
        flag = 0
        A_mat = []
        list_rand = random.sample(range(0, rand_numb), 8)
        for i in list_rand:
            A_temp = [pts1[i,0]*pts2[i,0], pts1[i,1]*pts2[i,0], pts2[i,0], pts1[i,0]*pts2[i,1], pts1[i,1]*pts2[i,1], pts2[i,1], pts1[i,0], pts1[i,1], 1.0]
            A_mat.append(A_temp) 
        A_mat = np.array(A_mat)
        fn = null_space(A_mat)[:,0]
        f_mat = fn.reshape((3,3))
        f_mat = f_mat/f_mat[2,2]
        [U,D,V] = svd(f_mat)
        D[2] = 0
        mod_f = U@np.diag(D) @ V
        for i in range(0, rand_numb):
            u = np.array([pts1[i,0],pts1[i,1],1])
            v = np.array([pts2[i,0],pts2[i,1],1])
            err = (v.T) @ mod_f @ u
            if(np.abs(err) < threshold):
                flag = flag + 1
        inlines[iter_numb] = flag
        F_matrix.append(mod_f)
        iter_numb = iter_numb + 1
    idx = np.argmax(inlines)
    F = F_matrix[idx]
'''

'''
# Trial to check projection matrix
Trial_1 = Proj_mat @ np.array([[all_data_dt[1,4]], [all_data_dt[1,5]], [1]])
lamb = 1/Trial_1[2,0]
Trial_2 = lamb * Trial_1
'''

# Trial to check projection matrix
dx1 = all_data_dt[0, 0]
dy1 = all_data_dt[0, 1]
dx2 = all_data_dt[0, 2]
dy2 = all_data_dt[0, 3]
dX = all_data_dt[0, 4]
dY = all_data_dt[0, 5]
Pix_diff = np.array([[0, -1, dy1], [1, 0, -dx1], [-dy1, dx1, 0], [0, -1, dy2], [1, 0, -dx2], [-dy2, dx2, 0]])
Trial_3 = Pix_diff @ final_proj_matrix
P_sol, D_sol, Q_sol = np.linalg.svd(Trial_3, full_matrices=True)
Sol = np.array([[Q_sol[2, 0]], [Q_sol[2, 1]], [Q_sol[2, 2]]])
lamb = 1/Sol[2, 0]
Sol_2 = lamb * Sol

#A = np.array([all_data_dt[1,0], all_data_dt[1,1], 1])
#B = np.array([all_data_dt[1,2], all_data_dt[1,3], 1])
#line_A = Fund_mat_clean @ B
#line_B = Fund_mat_clean.T @ A
