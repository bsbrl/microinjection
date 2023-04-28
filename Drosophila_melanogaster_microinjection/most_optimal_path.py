# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 09:23:27 2021

@author: enet-joshi317-admin
"""

from ML.transformation_matrix_DSLR_4x import function_transformation_matrix_DSLR_4x
import cv2
import numpy as np
from optimal_ee_5940.Piecewise_Optimal_Path import original_path_func,optimal_path_func,optimal_path_complete_func,optimal_path_func_box
import math
# Initial Variables
d=0
im_width=400
im_height=400
V=math.hypot(10000,10000)
inv_V=(float(float(1)/float(V)))
distance_traveled_dishes=[]

def most_optimal_path(filename,y1a_rc,y2a_rc,x1a_rc,x2a_rc):

    x_state_list_ends_orig_list=[]
    y_state_list_ends_orig_list=[]   
    img_whole_dish_path=cv2.imread('C:/Users/me-alegr011-admin/Downloads/Robot_code/DSLR_Camera/'+filename,1)
    e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep=original_path_func(img_whole_dish_path,y1a_rc,y2a_rc,x1a_rc,x2a_rc,im_width,im_height,filename)
# #    # Paths calculated
#     original_path_total=range(0,len(e_count_list))
#     original_path_complete,x_state_list_orig,y_state_list_orig,x_state_list_ends_orig,y_state_list_ends_orig=optimal_path_complete_func(original_path_total,'Original',img_whole_dish_path,e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep,filename)
#     x_state_list_ends_orig_list.append(x_state_list_ends_orig)
#     y_state_list_ends_orig_list.append(y_state_list_ends_orig)
#     print('Done Original')
# #    img_whole_dish_path=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
# #    optimal_path_total_max_x=optimal_path_func('Max x',e_count_list,xc_rc_keep,yc_rc_keep)
# #    optimal_path_complete_max_x,x_state_list_orig,y_state_list_orig,x_state_list_ends_orig,y_state_list_ends_orig=optimal_path_complete_func(optimal_path_total_max_x,'Max x',img_whole_dish_path,e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep,filename)
# #    x_state_list_ends_orig_list.append(x_state_list_ends_orig)
# #    y_state_list_ends_orig_list.append(y_state_list_ends_orig)
# #    print('Done Max x')
# #    img_whole_dish_path=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
# #    optimal_path_total_min_x=optimal_path_func('Min x',e_count_list,xc_rc_keep,yc_rc_keep)
# #    optimal_path_complete_min_x,x_state_list_orig,y_state_list_orig,x_state_list_ends_orig,y_state_list_ends_orig=optimal_path_complete_func(optimal_path_total_min_x,'Min x',img_whole_dish_path,e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep,filename)
# #    x_state_list_ends_orig_list.append(x_state_list_ends_orig)
# #    y_state_list_ends_orig_list.append(y_state_list_ends_orig)
# #    print('Done Min x')
# #    img_whole_dish_path=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
# #    optimal_path_total_max_y=optimal_path_func('Max y',e_count_list,xc_rc_keep,yc_rc_keep)
# #    optimal_path_complete_max_y,x_state_list_orig,y_state_list_orig,x_state_list_ends_orig,y_state_list_ends_orig=optimal_path_complete_func(optimal_path_total_max_y,'Max y',img_whole_dish_path,e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep,filename)
# #    x_state_list_ends_orig_list.append(x_state_list_ends_orig)
# #    y_state_list_ends_orig_list.append(y_state_list_ends_orig)
# #    print('Done Max y')
# #    img_whole_dish_path=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
# #    optimal_path_total_min_y=optimal_path_func('Min y',e_count_list,xc_rc_keep,yc_rc_keep)
# #    optimal_path_complete_min_y,x_state_list_orig,y_state_list_orig,x_state_list_ends_orig,y_state_list_ends_orig=optimal_path_complete_func(optimal_path_total_min_y,'Min y',img_whole_dish_path,e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep,filename)
# #    x_state_list_ends_orig_list.append(x_state_list_ends_orig)
# #    y_state_list_ends_orig_list.append(y_state_list_ends_orig)
# #    print('Done Min y')
# #    img_whole_dish_path=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
# #    optimal_path_total_regular=optimal_path_func('Regular',e_count_list,xc_rc_keep,yc_rc_keep)
# #    optimal_path_complete_regular,x_state_list_orig,y_state_list_orig,x_state_list_ends_orig,y_state_list_ends_orig=optimal_path_complete_func(optimal_path_total_regular,'Regular',img_whole_dish_path,e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep,filename)
# #    x_state_list_ends_orig_list.append(x_state_list_ends_orig)
# #    y_state_list_ends_orig_list.append(y_state_list_ends_orig)
# #    print('Done Regular')
# #    img_whole_dish_path=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
# #    optimal_path_total_box=optimal_path_func_box(e_count_list,xc_rc_keep,yc_rc_keep)
# #    optimal_path_complete_box,x_state_list_orig,y_state_list_orig,x_state_list_ends_orig,y_state_list_ends_orig=optimal_path_complete_func(optimal_path_total_box,'Box',img_whole_dish_path,e_count_list,xc_rc_keep,yc_rc_keep,y1_rc_keep,x1_rc_keep,y2_rc_keep,x2_rc_keep,filename)
# #    x_state_list_ends_orig_list.append(x_state_list_ends_orig)
# #    y_state_list_ends_orig_list.append(y_state_list_ends_orig)
# #    print('Done Box')
# #    paths=[optimal_path_complete_box]
# #    path_names=['Box']
#     paths=[original_path_complete]
#     path_names=['Original']
# #    paths=[original_path_complete,optimal_path_complete_max_x,optimal_path_complete_min_x,optimal_path_complete_max_y,optimal_path_complete_min_y,optimal_path_complete_regular,optimal_path_complete_box]
# #    path_names=['Original','Max_x','Min_x','Max_y','Min_y','Regular','Box']
# #    img_whole_dish=cv2.imread('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/DSLR_Camera/'+filename,1)
# #    emb=0
# #    xc_e_list=[]
# #    yc_e_list=[]
# #    for i in paths[0]:
# #        print('Embryo {} out of {} Embryos'.format(emb+1,len(paths[0])))
# #        crop_img_embryo=img_whole_dish[y_state_list_ends_orig_list[0][i][0]:y_state_list_ends_orig_list[0][i][1],x_state_list_ends_orig_list[0][i][0]:x_state_list_ends_orig_list[0][i][1]]
# #        cv2.imwrite('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/Embryo_DSLR_Dish/embryo_{}.jpg'.format(emb+1),crop_img_embryo)
# #        cX_save_old_orig_total,y_assoc_min_old_orig_total,embryo_length_y,xc_e,yc_e,x_inj_post,y_inj_post=injection_prediction_algo_new_new_cameras_centroid('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/Embryo_DSLR_Dish/embryo_{}.jpg'.format(emb+1),0,y_state_list_ends_orig_list[0][i][1]-y_state_list_ends_orig_list[0][i][0],0,x_state_list_ends_orig_list[0][i][1]-x_state_list_ends_orig_list[0][i][0],0)             
# #        xc_e_list.append(xc_e)
# #        yc_e_list.append(yc_e)
# #        emb+=1
#     X_Y_dishes=[]
#     for g in range(len(paths)):
#         total_distance=0
#         emb=0
#         positions=[]
#         print('Path {}'.format(g+1))
#         for i in paths[g]:
#             embryo_point_center = function_transformation_matrix_DSLR_4x(np.mean([x_state_list_ends_orig_list[g][i][0],x_state_list_ends_orig_list[g][i][1]]),np.mean([y_state_list_ends_orig_list[g][i][0],y_state_list_ends_orig_list[g][i][1]]),1860,2817,3041,1433,1038,1204,66833,92088,37453,66298,32103,109048) # 4x 
# #            embryo_point_center = function_transformation_matrix_DSLR_4x(xc_e_list[i]+x_state_list_ends_orig_list[g][i][0],yc_e_list[i]+y_state_list_ends_orig_list[g][i][0],1744,865,1294,2309,2927,2539,25413,93278,56083,103368,61273,68508) # 4x 
#             embryo_point_center = embryo_point_center + np.matrix([[0],[0]])
#             positions.append([int(float(embryo_point_center.item(0,0))),int(float(embryo_point_center.item(1,0)))])
#             if emb==0:
#                 emb+=1
#             else:
#                 total_distance+=math.hypot(positions[emb-1][0]-positions[emb][0],positions[emb-1][1]-positions[emb][1])
#                 emb+=1
#         distance_traveled_dishes.append(total_distance)
#         X_Y_dishes.append(positions)
#     min_dist_traveled=min(distance_traveled_dishes)
#     index_min=distance_traveled_dishes.index(min_dist_traveled)
#     x1a_rc_post=[row[0] for row in x_state_list_ends_orig_list[index_min]]
#     y1a_rc_post=[row[0] for row in y_state_list_ends_orig_list[index_min]]
#     x2a_rc_post=[row[1] for row in x_state_list_ends_orig_list[index_min]]
#     y2a_rc_post=[row[1] for row in y_state_list_ends_orig_list[index_min]]
#     print('Most Optimal Path = '+path_names[index_min])
    return x1_rc_keep,y1_rc_keep,x2_rc_keep,y2_rc_keep,xc_rc_keep,yc_rc_keep
#filename='Entire_Petri_Dish_331.jpg'
#image=1
#y1a_rc=np.load('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/petri_dishes_optimal_path/y1a_rc_{}.npy'.format(image+1))
#x1a_rc=np.load('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/petri_dishes_optimal_path/x1a_rc_{}.npy'.format(image+1))
#y2a_rc=np.load('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/petri_dishes_optimal_path/y2a_rc_{}.npy'.format(image+1))
#x2a_rc=np.load('C:/Users/enet-joshi317-admin/Downloads/Amey Code-20190710T183750Z-001/Amey Code/petri_dishes_optimal_path/x2a_rc_{}.npy'.format(image+1))
#final_X_Y,x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post=most_optimal_path(filename,y1a_rc,y2a_rc,x1a_rc,x2a_rc)