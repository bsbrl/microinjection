# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 13:40:09 2021

@author: enet-joshi317-admin
"""
from ML.sorting import sorting
def order(xc_rc,yc_rc,missed_embryos,num_order):
    if num_order==0:
        y1a_rc_new=[]
        y2a_rc_new=[]
        x1a_rc_new=[]
        x2a_rc_new=[]
        count_list=[]
        for i in range(10):
            for j in range(15):
                for k in range(len(xc_rc)):
                    if i*400<yc_rc[k][0]<=(i+1)*400 and j*400<xc_rc[k][0]<=(j+1)*400 and xc_rc[k] not in count_list:
                        y1a_rc_new.append([int(yc_rc[k][1]),yc_rc[k][3],yc_rc[k][4],1])
                        y2a_rc_new.append([int(yc_rc[k][2]),yc_rc[k][3],yc_rc[k][4],1])
                        x1a_rc_new.append([int(xc_rc[k][1]),xc_rc[k][3],xc_rc[k][4],1])
                        x2a_rc_new.append([int(xc_rc[k][2]),xc_rc[k][3],xc_rc[k][4],1])
                        count_list.append(xc_rc[k])
    else:
        y1a_rc=[]
        y2a_rc=[]
        x1a_rc=[]
        x2a_rc=[]
        count_list=[]
        for i in range(10):
            for j in range(15):
                for k in range(len(xc_rc)):
                    if i*400<yc_rc[k][0]<=(i+1)*400 and j*400<xc_rc[k][0]<=(j+1)*400 and xc_rc[k] not in count_list:
                        y1a_rc.append([int(yc_rc[k][1]),yc_rc[k][3],yc_rc[k][4],1])
                        y2a_rc.append([int(yc_rc[k][2]),yc_rc[k][3],yc_rc[k][4],1])
                        x1a_rc.append([int(xc_rc[k][1]),xc_rc[k][3],xc_rc[k][4],1])
                        x2a_rc.append([int(xc_rc[k][2]),xc_rc[k][3],xc_rc[k][4],1])
                        count_list.append(xc_rc[k])
        same_row=[]
        greater_than=[]
        x1a_rc_old=x1a_rc
        y1a_rc_old=y1a_rc
        x2a_rc_old=x2a_rc
        y2a_rc_old=y2a_rc
        for i in range(missed_embryos):
            for j in range(len(x1a_rc)):
                if x1a_rc[j][1]==x1a_rc_old[len(x1a_rc_old)-1-i][1]:
                    same_row.append(x1a_rc[j][0])
                else:
                    continue
            for k in range(len(same_row)):
                if same_row[k]>x1a_rc_old[len(x1a_rc_old)-1-i][1]:
                    greater_than.append(same_row[k])
                else:
                    continue
            just_x=[]
            for q in range(len(x1a_rc)):
                just_x.append(x1a_rc[q][0])
            if greater_than!=[]:
                min_greater=min(greater_than)
                replace=just_x.index(min_greater)
                x1a_rc[replace][0],x1a_rc_old[len(x1a_rc_old)-1-i][0]=x1a_rc_old[len(x1a_rc_old)-1-i][0],x1a_rc[replace][0]
                y1a_rc[replace][0],y1a_rc_old[len(x1a_rc_old)-1-i][0]=y1a_rc_old[len(x1a_rc_old)-1-i][0],y1a_rc[replace][0]
                x2a_rc[replace][0],x2a_rc_old[len(x1a_rc_old)-1-i][0]=x2a_rc_old[len(x1a_rc_old)-1-i][0],x2a_rc[replace][0]
                y2a_rc[replace][0],y2a_rc_old[len(x1a_rc_old)-1-i][0]=y2a_rc_old[len(x1a_rc_old)-1-i][0],y2a_rc[replace][0]
    
                x1a_rc[replace][1],x1a_rc_old[len(x1a_rc_old)-1-i][1]=x1a_rc_old[len(x1a_rc_old)-1-i][1],x1a_rc[replace][1]
                y1a_rc[replace][1],y1a_rc_old[len(x1a_rc_old)-1-i][1]=y1a_rc_old[len(x1a_rc_old)-1-i][1],y1a_rc[replace][1]
                x2a_rc[replace][1],x2a_rc_old[len(x1a_rc_old)-1-i][1]=x2a_rc_old[len(x1a_rc_old)-1-i][1],x2a_rc[replace][1]
                y2a_rc[replace][1],y2a_rc_old[len(x1a_rc_old)-1-i][1]=y2a_rc_old[len(x1a_rc_old)-1-i][1],y2a_rc[replace][1]
                
                x1a_rc[replace][2],x1a_rc_old[len(x1a_rc_old)-1-i][2]=x1a_rc_old[len(x1a_rc_old)-1-i][2],x1a_rc[replace][2]
                y1a_rc[replace][2],y1a_rc_old[len(x1a_rc_old)-1-i][2]=y1a_rc_old[len(x1a_rc_old)-1-i][2],y1a_rc[replace][2]
                x2a_rc[replace][2],x2a_rc_old[len(x1a_rc_old)-1-i][2]=x2a_rc_old[len(x1a_rc_old)-1-i][2],x2a_rc[replace][2]
                y2a_rc[replace][2],y2a_rc_old[len(x1a_rc_old)-1-i][2]=y2a_rc_old[len(x1a_rc_old)-1-i][2],y2a_rc[replace][2]
            else:
                continue
        separate=[]
        separate_all=[]
        just_x1=[]
        for q in range(len(x1a_rc)):
            just_x1.append(x1a_rc[q][0])
        just_y1=[]
        for q in range(len(y1a_rc)):
            just_y1.append(y1a_rc[q][0])
        just_x2=[]
        for q in range(len(x2a_rc)):
            just_x2.append(x2a_rc[q][0])
        just_y2=[]
        for q in range(len(y2a_rc)):
            just_y2.append(y2a_rc[q][0])
            
        for i in range(len(x1a_rc)):
            if x1a_rc[i][0]==x1a_rc[len(x1a_rc)-1][0]:
                if x1a_rc[i][1]==x1a_rc[i-1][1]:
                    separate.append(x1a_rc[i][0])
                    separate_all.append(separate)
                else:
                    separate_all.append(separate)
                    separate=[]
                    separate.append(x1a_rc[i][0])
                    separate_all.append(separate)              
            elif x1a_rc[i][1]==x1a_rc[i+1][1]:
                separate.append(x1a_rc[i][0])
            else:
                if separate==[]:
                    separate.append(x1a_rc[i][0])
                    separate_all.append(separate)
                elif x1a_rc[i][1]==x1a_rc[i-1][1]:
                    separate.append(x1a_rc[i][0])
                    separate_all.append(separate)                
                else:
                    separate_all.append(separate)
                separate=[]
        y1a_rc_new=[]
        y2a_rc_new=[]
        x1a_rc_new=[]
        x2a_rc_new=[]
        for i in range(len(separate_all)):
            sorted_list=(sorting(separate_all[i]))
            for j in range(len(sorted_list)):
                x1a_rc_new.append([x1a_rc[just_x1.index(sorted_list[j])][0],x1a_rc[just_x1.index(sorted_list[j])][1],x1a_rc[just_x1.index(sorted_list[j])][2],x1a_rc[just_x1.index(sorted_list[j])][3]])
                y1a_rc_new.append([y1a_rc[just_x1.index(sorted_list[j])][0],y1a_rc[just_x1.index(sorted_list[j])][1],y1a_rc[just_x1.index(sorted_list[j])][2],y1a_rc[just_x1.index(sorted_list[j])][3]])
                x2a_rc_new.append([x2a_rc[just_x1.index(sorted_list[j])][0],x2a_rc[just_x1.index(sorted_list[j])][1],x2a_rc[just_x1.index(sorted_list[j])][2],x2a_rc[just_x1.index(sorted_list[j])][3]])
                y2a_rc_new.append([y2a_rc[just_x1.index(sorted_list[j])][0],y2a_rc[just_x1.index(sorted_list[j])][1],y2a_rc[just_x1.index(sorted_list[j])][2],y2a_rc[just_x1.index(sorted_list[j])][3]])

    return y1a_rc_new,y2a_rc_new,x1a_rc_new,x2a_rc_new
            
