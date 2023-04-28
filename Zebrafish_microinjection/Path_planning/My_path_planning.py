# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:28:07 2021

@author: admin
"""


import numpy as np
import time
import cv2
import itertools

class My_path_planning():
    def __init__(self, box_center, box_coordinate, class_ids):
        flag = 1
        self.box_center = box_center
        self.box_coordinate = box_coordinate
        self.class_ids = class_ids
        
    def total_dis(self, path, box_center):
        len_path = len(path)
        total_distance = 0
        for i in range(len_path-1):
            total_distance = total_distance + self.dis(path[i], path[i+1], box_center)
        return total_distance
    
    def dis(self, first, second, box_center):
        distance = np.sqrt(np.square(box_center[first][0] - box_center[second][0]) + np.square(box_center[first][1] - box_center[second][1]))
        return distance
    
    def image_with_path(self, path, image, image_name, box_center):
        len_path = len(path)
        for i in range(len_path-1):
            image = cv2.line(image, (int(box_center[path[i]][0]), int(box_center[path[i]][1])), (int(box_center[path[i+1]][0]), int(box_center[path[i+1]][1])) , (0, 0, 255), 3)
        cv2.imwrite(image_name, image)
        
    def factorial(self, n):
        if n == 1:
            return n
        else:
            return n * factorial(n-1)
        
    def path_to_box_center_coordinate(self, path, orig_box_center, orig_box_coordinate, orig_class_ids):
        new_box_center = []
        new_box_coordinate = []
        new_class_ids = []
        for i in range(len(path)):
            new_box_center.append(orig_box_center[path[i]])
            new_box_coordinate.append(orig_box_coordinate[path[i]])
            new_class_ids.append(orig_class_ids[path[i]])
        new_box_center = np.asarray(new_box_center)
        new_box_coordinate = np.asarray(new_box_coordinate)
        new_class_ids = np.asarray(new_class_ids)
        return new_box_center, new_box_coordinate, new_class_ids
    
    def Greedy_path(self, box_center, box_coordinate, class_ids):
        image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_detection.jpg')
        length = len(box_center)
        length = len(box_center)
        start_time_greedy = time.time()
        update_box_center = box_center
        current_point = np.array((0, 0))
        sorted_box_center = []
        sorted_box_coordinate = []
        total_dis = []
        total_path_greedy = []
        
        for i in range(length):
            updated_length = len(update_box_center)
            dis = []
            for j in range(0, updated_length):
                dis.append(np.sqrt(np.square(current_point[0] - update_box_center[j][0]) + np.square(current_point[1] - update_box_center[j][1])))
            min_dis = min(dis)
            total_dis.append(min_dis)
            min_index = dis.index(min(dis))
            current_point = update_box_center[min_index]
            update_box_center = np.delete(update_box_center, np.where((update_box_center[:,0] == current_point[0]) & (update_box_center[:,1] == current_point[1])), 0)
            path_greedy = np.where((box_center[:,0] == current_point[0]) & (box_center[:,1] == current_point[1]))[0][0]
            total_path_greedy.append(path_greedy)
            sorted_box_center.append(current_point)
        
        sorted_box_center = np.asarray(sorted_box_center)
        sum_dis_greedy = sum(total_dis)
        # print('Greedy algo total dis', sum_dis_greedy)
        # print('Time for greedy force algo', (time.time() - start_time_greedy), 'sec')
        # print('Greedy algo optimal path', total_path_greedy)
        # print('Done with Greedy algo \n')
        # image_with_path(total_path_greedy, image, 'DSLR_image_greedy_path.jpg', box_center)
        sorted_box_center, sorted_box_coordinate, sorted_class_ids = self.path_to_box_center_coordinate(total_path_greedy, box_center, box_coordinate, class_ids)
        return sorted_box_center, sorted_box_coordinate, sorted_class_ids, total_path_greedy
    
    
    def Greedy_dynamic(self, box_center, box_coordinate):
        image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_detection.jpg')
        box_center, box_coordinate, class_ids, total_path_greedy = self.Greedy_path(box_center, box_coordinate, class_ids)
        update_greedy_path = total_path_greedy
        update_path_len = len(update_greedy_path)
        
        total_distance_2 = self.total_dis(update_greedy_path, box_center)
        
        future_states = 5
        start_time_greedy_brut = time.time()
        for i in range(0, update_path_len - future_states):
            sudo_greedy_path = update_greedy_path.copy()
            total_j_dis = []
            all_permutations = list(itertools.permutations(range(1,future_states)))
            for j in all_permutations:
                # print(j)
                for k in range(len(j)):
                    sudo_greedy_path[i+k+1] = update_greedy_path[i+j[k]]
                j_dis = self.total_dis(sudo_greedy_path, box_center)
                total_j_dis.append(j_dis)
            min_dis = min(total_j_dis)
            index_min_dis = total_j_dis.index(min(total_j_dis))
            min_sequency = all_permutations[index_min_dis]
            for k in range(len(min_sequency)):
                sudo_greedy_path[i+k+1] = update_greedy_path[i+min_sequency[k]]
            update_greedy_path = sudo_greedy_path.copy()
            # print(total_j_dis)
            # print(min_dis)
            # print(index_min_dis)
            # print(sudo_greedy_path)
            # print(update_greedy_path)
                # for k in list(j):
                    # print(update_greedy_path[i+k])
        
        # print('Greedy algo with brut force total dis', min_dis)
        # print('Time for greedy and brut force algo', (time.time() - start_time_greedy_brut), 'sec')
        # print('Greedy algo with dynamic path', update_greedy_path)
        # image_with_path(update_greedy_path, image, 'DSLR_image_greedy_dynamic_path.jpg', box_center)
        # print('Done greedy with dynamic \n')
        sorted_box_center, sorted_box_coordinate, sorted_class_ids = self.path_to_box_center_coordinate(update_greedy_path, box_center, box_coordinate, class_ids)
        return sorted_box_center, sorted_box_coordinate, update_greedy_path
    
    
    def Greedy_opt2(self, box_center, box_coordinate, class_ids):
        sorted_box_center, sorted_box_coordinate, sorted_class_ids, total_path_greedy = self.Greedy_path(box_center, box_coordinate, class_ids)
        image = cv2.imread('D:/Microinjection_Project/Python_Code/DSLR_image_detection.jpg')
        #___________________________#
        ######
        # Greedy plus 2-opt path #
        #'''
        start_time_opt2 = time.time()
        opt2_path = total_path_greedy
        len_2opt = len(box_center)
        
        class Point:
        	def __init__(self,x,y):
        		self.x = x
        		self.y = y
        
        def ccw(A,B,C):
            return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
        
        def intersect(A,B,C,D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
        def reverse(lst, start, stop):
            new_lst = lst.copy()
            ranges_2 = range(-(stop-start), (stop-start)+2, 2)
            flag = 0
            for i in range(start, stop+1):
                new_lst[i] = lst[i - ranges_2[flag]]
                flag = flag + 1
            return new_lst
        
        def opt2_algo(path):
            len_2opt = len(path)
            restart = 1
            while restart == 1:
                restart = 0
                for i in range(1, len_2opt-1):
                    # print(i)
                    for j in range(0, i-1):
                        a = Point(box_center[path[i]][0], box_center[path[i]][1])
                        b = Point(box_center[path[i+1]][0], box_center[path[i+1]][1])
                        c = Point(box_center[path[j]][0], box_center[path[j]][1])
                        d = Point(box_center[path[j+1]][0], box_center[path[j+1]][1])
                        if intersect(a,b,c,d) == True:
                            # print('i =',i,'j =',j, intersect(a,b,c,d))
                            new_path = reverse(path, j+1, i)
                            path = new_path.copy()
                            restart = 1
                            break
                    else:
                        continue
                    break
            return path
            
                        
        opt2_path = opt2_algo(opt2_path)
        # print('2-opt algo total dis', total_dis(opt2_path, box_center))
        # print('Time for 2-opt algo', (time.time() - start_time_opt2), 'sec')
        print('Optimal path', opt2_path)
        self.image_with_path(opt2_path, image, 'DSLR_image_2opt_path.jpg', box_center)
        print('Done with Path planning \n')
        sorted_box_center, sorted_box_coordinate, sorted_class_ids = self.path_to_box_center_coordinate(opt2_path, box_center, box_coordinate, class_ids)
        return sorted_box_center, sorted_box_coordinate, sorted_class_ids, opt2_path

# box_center = np.load('D:/Microinjection_Project/Python_Code/box_center.npy')
# box_coordinate = np.load('D:/Microinjection_Project/Python_Code/box_coordinate.npy')
# class_ids = np.load('D:/Microinjection_Project/Python_Code/class_ids.npy')
# Path = My_path_planning(box_center, box_coordinate, class_ids)
# sorted_box_center, sorted_box_coordinate, sorted_class_ids, total_path_greedy = Path.Greedy_opt2(box_center, box_coordinate, class_ids)

