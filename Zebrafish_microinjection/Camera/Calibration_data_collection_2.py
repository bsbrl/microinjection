# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:39:57 2022

@author: admin
"""


import numpy as np
import pandas as pd
from scipy.linalg import null_space
from numpy.linalg import matrix_rank
import cv2
import matplotlib.pyplot as plt


# Reading xyz stage data and yolk pixel data from save images and microinjection data
xyz_data = np.load("C:/Users/asjos/Downloads/03-30-2022_11-33-34/change_xyz.npy", allow_pickle=True)
yolk_data = np.load("C:/Users/asjos/Downloads/03-30-2022_11-33-34/yolk_data.npy", allow_pickle=True)

# Arrange data in required order to save
all_data = np.empty([1, 7])
for i in range(len(xyz_data)):
    for j in range(len(xyz_data[i])):
        current_row = np.array([yolk_data[i][j][0], yolk_data[i][j][1], yolk_data[i][j][2], yolk_data[i][j][3], xyz_data[i][j][0], xyz_data[i][j][1], xyz_data[i][j][2]])
        # current_row = np.array([yolk_data[i][j][0], yolk_data[i][j][1], yolk_data[i][j][2], yolk_data[i][j][3]])
        all_data = np.vstack([all_data, current_row])
        # all_data.append(current_row)

# Save data as required 
all_data = np.delete(all_data, (0), axis=0)
pd.DataFrame(all_data).to_csv("Calibration_data_collection_2.csv")
# np.savetxt('Calibration_data_collection.csv', all_data, fmt='%s')

# Changing delta x, y, z location to its corresponding pixel co ordinates
all_data_new = np.copy(all_data)
all_data_new[1:len(all_data_new), 4:7] = all_data_new[0:(len(all_data_new)-1), 4:7]
all_data_new[0, 4:7] = 0

for i in range(len(all_data_new)):
    if all_data_new[i, 6] != 0:
        all_data_new[i, 4:7] = 0 
        
# Changingin pixel co-ordinates to delta pixel co-ordinates 
all_data_correct = np.copy(all_data_new)
basic_pixel = all_data_correct[0]
for i in range(len(all_data_correct)):
    if all_data_new[i, 4] == 0:
        basic_pixel = all_data_new[i]
        print(basic_pixel)
    else:
        basic_pixel = all_data_new[i-1]
        all_data_correct[i, 0:4] = all_data_new[i, 0:4] - basic_pixel[0:4]
        
# Removing first pixel locations
all_data_dt = np.copy(all_data_correct)
all_data_dt = np.delete(all_data_dt, np.where(all_data_correct[:,4] == 0), 0)

# Save npy value and cvs file
pd.DataFrame(all_data_dt).to_csv("Calibration_data_collection_ProjectionMatrix.csv")
np.save('Calibration_data_collection_ProjectionMatrix.npy', all_data_dt)


'''
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
'''