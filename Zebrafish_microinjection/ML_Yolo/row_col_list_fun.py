# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 00:12:47 2020

@author: Andrew
"""
from os import listdir
from os.path import isfile, join

def row_col_list_fun(onlyfiles):
    row=[]
    col=[]
    for i in range(len(onlyfiles)):
        if onlyfiles[i][len(onlyfiles[i])-6]=='_' and onlyfiles[i][len(onlyfiles[i])-8]=='_':
            row.append(int(onlyfiles[i][len(onlyfiles[i])-7]))
            col.append(int(onlyfiles[i][len(onlyfiles[i])-5]))
        elif onlyfiles[i][len(onlyfiles[i])-7]=='_' and onlyfiles[i][len(onlyfiles[i])-9]=='_':
            row.append(int(onlyfiles[i][len(onlyfiles[i])-8]))
            col.append(int(onlyfiles[i][len(onlyfiles[i])-6:len(onlyfiles[i])-4]))
        elif onlyfiles[i][len(onlyfiles[i])-7]!='_' and onlyfiles[i][len(onlyfiles[i])-8]!='_' and onlyfiles[i][len(onlyfiles[i])-5]!='_' and onlyfiles[i][len(onlyfiles[i])-6]=='_':
            row.append(int(onlyfiles[i][len(onlyfiles[i])-8:len(onlyfiles[i])-6]))
            col.append(int(onlyfiles[i][len(onlyfiles[i])-5]))
        else:
            row.append(int(onlyfiles[i][len(onlyfiles[i])-9:len(onlyfiles[i])-7]))
            col.append(int(onlyfiles[i][len(onlyfiles[i])-6:len(onlyfiles[i])-4]))
    return row,col
# onlyfiles=[f for f in listdir('C:/Users/Andrew/anaconda3/Example_of_Annotations_Folder/test_images') if isfile(join('C:/Users/Andrew/anaconda3/Example_of_Annotations_Folder/test_images', f))]
# row,col=row_col_list_fun(onlyfiles)