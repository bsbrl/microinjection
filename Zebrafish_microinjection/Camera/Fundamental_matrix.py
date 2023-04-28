# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 09:52:20 2022

@author: admin
"""


import numpy as np
import pandas as pd
from scipy.linalg import null_space
from numpy.linalg import matrix_rank
import cv2
import matplotlib.pyplot as plt

## Finding fundamental matrix in this code from all data read from calibration data all

# Reading calibration data all
all_data = pd.read_excel('Calibration_data_all.xlsx')
# all_data = pd.read_csv('Calibration_data_collection.csv')
all_data = all_data.to_numpy()
all_data = np.delete(all_data, 0, 1)

# Finding null space here
matrix = np.empty([1, 9])
for i in range(len(all_data)):
    x_11 = all_data[i][0]
    x_21 = all_data[i][2]
    y_11 = all_data[i][1]
    y_21 = all_data[i][3]
    current_row = np.array([x_11*x_21, x_11*y_21, x_11, x_21*y_11, y_11*y_21, y_11, x_21, y_21, 1])
    matrix = np.vstack([matrix, current_row])
    
matrix = np.delete(matrix, (0), axis=0)
matrix = np.asmatrix(matrix)
P, D, Q = np.linalg.svd(matrix, full_matrices=True)
# null_s = null_space(matrix)
# rank = matrix_rank(matrix)
# null_s * np.sign(null_s[0,0])

Fund_mat = np.array([[Q[8,0], Q[8,1], Q[8,2]], [Q[8,3], Q[8,4], Q[8,5]], [Q[8,6], Q[8,7], Q[8,8]]])
rank = matrix_rank(Fund_mat)
P_fund, D_fund, Q_fund = np.linalg.svd(Fund_mat, full_matrices=True)
D_fund_clean = np.diag(D_fund)
D_fund_clean[2,2] = 0
Fund_mat_clean = P_fund @ D_fund_clean @ Q_fund
rank_clean = matrix_rank(Fund_mat_clean)

# Trial to check fundamental matrix
A = np.array([all_data[1,0], all_data[1,1], 1])
B = np.array([all_data[1,2], all_data[1,3], 1])
line_A = Fund_mat_clean @ B
line_B = Fund_mat_clean.T @ A

# To check if Fundamental matrix and line equations are correct. All solution values should be close to zero
solution = np.transpose(A) @ Fund_mat_clean @ B 
solution_A = A.T @ line_A
solution_B = B.T @ line_B

img = cv2.imread('D:/Microinjection_Project/Python_Code/Injection_Images/03-28-2022_11-33-44/Embryo_left_7_2.jpg')
x = np.linspace(0,1280,1281)
# y = line_A[0]*x + line_A[1]*y + line_A[2]
y = (-line_A[0]*x - line_A[2])/line_A[1]
plot1 = plt.figure(1)
plt.plot(x, y)
plt.plot(A[0], A[1], 'r+')
plt.imshow(img)

img = cv2.imread('D:/Microinjection_Project/Python_Code/Injection_Images/03-28-2022_11-33-44/Embryo_right_7_2.jpg')
x = np.linspace(0,1280,1281)
# y = line_A[0]*x + line_A[1]*y + line_A[2]
y = (-line_B[0]*x - line_B[2])/line_B[1]
plot2 = plt.figure(2)
plt.plot(x, y)
plt.plot(B[0], B[1], 'r+')
plt.imshow(img)

plt.show()

# img = cv2.line(img, )


# import numpy as np
# X = np.random.normal(size=[20,18])
# P, D, Q = np.linalg.svd(X, full_matrices=True)
# X_a = P @ np.diag(D) @ Q
# print(np.std(X), np.std(X_a), np.std(X - X_a))
# print('Is X close to X_a?', np.isclose(X, X_a).all())