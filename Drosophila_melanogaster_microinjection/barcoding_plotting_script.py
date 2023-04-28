# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:47:56 2023

@author: User
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'size'   : 12}
matplotlib.rc('font', **font)
# change font
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

labels=['1','2','3','4','5','6','7','8']
a=np.load('<Load data path here>')
x=np.array([0,1,2,3,4,5,6,7,8,9,10])
# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a,x,rwidth=0.75)
ax.set_ylabel('Number of events')
ax.set_xlabel('Number of inserts')
ax.set_title('Barcode data')
plt.yticks(np.arange(0,18,1))
ax.set_xticklabels(labels)
ax.set_xticks(np.array([1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5]))
 
# Show plot
plt.show()
fig.savefig('barcode_data_plot_new.png', bbox_inches="tight")
fig.savefig('barcode_data_plot_new.pdf', bbox_inches="tight")
fig.savefig('barcode_data_plot_new.eps', bbox_inches="tight")