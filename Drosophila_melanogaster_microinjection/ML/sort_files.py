# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:44:53 2021

@author: User
"""

def sort_files(files):
    one_digit=[]
    two_digit=[]
    three_digit=[]
    for i in range(len(files)):
        if files[i][0].isdigit()==True and files[i][1].isdigit()==True and files[i][2].isdigit()==True:
            three_digit.append([int(files[i][0:3]),files[i]])
        elif files[i][0].isdigit()==True and files[i][1].isdigit()==True:
            two_digit.append([int(files[i][0:2]),files[i]])
        else:
            one_digit.append([int(files[i][0:1]),files[i]])
    
    three_digit.sort()
    two_digit.sort()
    one_digit.sort()
    files_final=one_digit+two_digit+three_digit
    return files_final
# from os import listdir
# from os.path import isfile, join
# path='C:/Users/User/Downloads/Andrew_files/Amey Code-20190710T183750Z-001/Amey Code/ML/ML_images_divided_bolus_paint'
# files = [ i for i in listdir(path) if isfile(join(path,i)) ]
# files_final=sort_files(files)