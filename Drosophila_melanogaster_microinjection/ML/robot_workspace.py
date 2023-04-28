# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:48:50 2022

@author: me-alegr011-admin
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

u, v = np.mgrid[0:1*np.pi:20j, 0:np.pi/2:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

# x = np.linspace(-10, 2,10 )
# y = np.linspace(-10, 10, 10)
# x, y = np.meshgrid(x, y)
# R=4
# z = np.sqrt(R**2 - x**2 - y**2)
fig = plt.figure()
 
# syntax for 3-D plotting
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(x, y, z)
ax.set_title('Robot workspace')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()