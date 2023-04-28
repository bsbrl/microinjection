# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 09:43:26 2022

@author: User
"""

def matrix_numbers(px_orig_1,py_orig_1,px_orig_2,py_orig_2,px1_1,py1_1,px1_2,py1_2,px2_1,py2_1,px2_2,py2_2,px3_1,py3_1,px3_2,py3_2,px4_1,py4_1,px4_2,py4_2):
    dpx1_1=px1_1-px_orig_1
    dpy1_1=py1_1-py_orig_1
    dpx1_2=px1_2-px_orig_2
    dpy1_2=py1_2-py_orig_2
    dpx2_1=px2_1-px_orig_1
    dpy2_1=py2_1-py_orig_1
    dpx2_2=px2_2-px_orig_2
    dpy2_2=py2_2-py_orig_2
    dpx3_1=px3_1-px_orig_1
    dpy3_1=py3_1-py_orig_1
    dpx3_2=px3_2-px_orig_2
    dpy3_2=py3_2-py_orig_2
    dpx4_1=px4_1-px_orig_1
    dpy4_1=py4_1-py_orig_1
    dpx4_2=px4_2-px_orig_2
    dpy4_2=py4_2-py_orig_2
    dp_list=[dpx1_1,dpy1_1,dpx1_2,dpy1_2,dpx2_1,dpy2_1,dpx2_2,dpy2_2,dpx3_1,dpy3_1,dpx3_2,dpy3_2,dpx4_1,dpy4_1,dpx4_2,dpy4_2]
    return dp_list

dp_list=matrix_numbers(560,323,748,330,523,279,723,322,534,288,707,333,570,307,749,309,538,292,734,322)
print(dp_list)