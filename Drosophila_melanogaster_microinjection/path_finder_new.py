# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:08:55 2022

@author: User
"""

from math import hypot
import cv2

def path_finder_new(x_list,y_list,x1_list,y1_list,x2_list,y2_list,img,filename):

    print(len(x_list))
    print(len(x1_list))
    
    x_list_new=[]
    y_list_new=[]
    while len(x_list)>0:
        max_x=max(x_list)
        min_y=min(y_list)
        index=x_list.index(max_x)
        y_max_x=y_list[index]
        if min_y+50>y_max_x>min_y-50:
            start_x_max_x=x_list[index]
            start_y_max_x=y_list[index]
            break
        else:
            x_list_new.append(x_list[index])
            y_list_new.append(y_list[index])
            del x_list[index]
            del y_list[index]
    x_list=x_list+x_list_new
    y_list=y_list+y_list_new
    x_list_new_new=[]
    y_list_new_new=[]
    while len(x_list)>0:
        min_x=min(x_list)
        min_y=min(y_list)
        index=x_list.index(min_x)
        y_min_x=y_list[index]
        if min_y+50>y_min_x>min_y-50:
            start_x_min_x=x_list[index]
            start_y_min_x=y_list[index]
            break
        else:
            x_list_new_new.append(x_list[index])
            y_list_new_new.append(y_list[index])
            del x_list[index]
            del y_list[index]
    x_list=x_list+x_list_new_new
    y_list=y_list+y_list_new_new
    if start_y_max_x<start_y_min_x:
        start_x=start_x_max_x
        start_y=start_y_max_x
    else:
        start_x=start_x_min_x
        start_y=start_y_min_x
    index=x_list.index(start_x)
    x_order=[start_x]
    y_order=[start_y]
    x1a_rc_post=[x1_list[index]]
    y1a_rc_post=[y1_list[index]]
    x2a_rc_post=[x2_list[index]]
    y2a_rc_post=[y2_list[index]]
    
    del x_list[index]
    del y_list[index]  
    del x1_list[index]
    del y1_list[index]  
    del x2_list[index]
    del y2_list[index]  
    
    while len(x_list)>0:
        dist=[]
        for i in range(len(x_list)):
            dist.append(hypot(x_order[len(x_order)-1]-x_list[i],y_order[len(y_order)-1]-y_list[i]))
        min_dist=min(dist)
        index=dist.index(min_dist)
        x_order.append(x_list[index])
        y_order.append(y_list[index])
        x1a_rc_post.append(x1_list[index])
        y1a_rc_post.append(y1_list[index])
        x2a_rc_post.append(x2_list[index])
        y2a_rc_post.append(y2_list[index])
        img_new=cv2.line(img, (x_order[len(x_order)-2], y_order[len(y_order)-2]), (x_order[len(x_order)-1], y_order[len(y_order)-1]), (0, 125, 0), thickness=3)
        del x_list[index]
        del y_list[index]
        del x1_list[index]
        del y1_list[index]  
        del x2_list[index]
        del y2_list[index]  
    cv2.imwrite('C:/Users/me-alegr011-admin/Downloads/Robot_code/Video images/Path_image/'+filename,img_new)

    return x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post,x_order,y_order

# import numpy as np
# # img=cv2.imread('C:/Users/User/Downloads/Entire_Petri_Dish_1024.jpg')
# img=cv2.imread('C:/Users/User/Downloads/Entire_Petri_Dish_1016.jpg')
# x=np.load('C:/Users/User/Downloads/xs_new.npy')
# y=np.load('C:/Users/User/Downloads/ys_new.npy')
# # x=np.load('C:/Users/User/Downloads/xs.npy')
# # y=np.load('C:/Users/User/Downloads/ys.npy')
# filename='Entire_Petri_Dish_1016.jpg'
# x_list=[]
# y_list=[]
# for i in range(len(x)):
#     x_list.append(int(x[i][0]))
#     y_list.append(int(y[i][0]))
# x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post,x_order,y_order=path_finder_new(x_list,y_list,img,filename)

# total_dist=[]
# for i in range(len(x_order)-1):
#     total_dist.append(hypot(x_order[i]-x_order[i+1],hypot(y_order[i]-y_order[i+1])))
# print('Total distance = ',np.sum(total_dist))

# x=np.load('C:/Users/User/Downloads/xs_old_new_path.npy')
# y=np.load('C:/Users/User/Downloads/ys_old_new_path.npy')
# # x=np.load('C:/Users/User/Downloads/xs_old_path.npy')
# # y=np.load('C:/Users/User/Downloads/ys_old_path.npy')
# x_list=[]
# y_list=[]
# for i in range(len(x)):
#     x_list.append(int(x[i]))
#     y_list.append(int(y[i]))
# total_dist=[]
# for i in range(len(x_list)-1):
#     total_dist.append(hypot(x_list[i]-x_list[i+1],hypot(y_list[i]-y_list[i+1])))
# print('Total distance = ',np.sum(total_dist)) 