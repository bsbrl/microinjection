# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:15:14 2022

@author: User
"""

import numpy as np
# from matplotlib import pyplot as plt

def pressure_calc(goal_sum_image,current_sum_image):
    Delta_sum_images=[0, 1559.5, 3893, 5624.5]
    Delta_pressures=[0,2,4,7]
    m,b=np.polyfit(Delta_sum_images,Delta_pressures,1)
    dp=-(m*(current_sum_image-goal_sum_image)+b)
    return dp
# dp=int(pressure_calc(10000,7000))
# print(dp)
# Delta_sum_images=[10662, 12222, 14555, 16287]
# Delta_pressures=[23,25,27,30]
# plt.figure(1)
# plt.title('Sum image vs pressure')
# plt.xlabel('x coordinate (px)')
# plt.ylabel('y coordinate (px)')
# plt.plot(Delta_pressures,Delta_sum_images,'bo',markersize=3)
# plt.show()