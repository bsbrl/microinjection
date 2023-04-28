# -*- coding: utf-8 -*-
"""
Created on Mon May 23 09:19:28 2022

@author: User
"""
# #%%
# import os
# os.system("taskkill /im python.exe /F")
#%%
from multiprocessing import Queue
from tkinter import Button,W,Tk,Label,Entry
import tkinter.font as font
from Main_code_function import Main_code_process,Stream_code_process,stream_function_multi_process,Main_code
from go_to_position_new import go_to_position_new
from PIL import ImageTk, Image
import time
import tkinter as tk
from multiprocessing import Process
import cv2
from XYZ_Stage.XYZ_Position import XYZ_Location
import serial

if __name__=='__main__':
    print('Starting')
    try:
      ser = serial.Serial(
        port="COM5",
        baudrate=9600,
      )
      ser.isOpen() # try to open port, if possible print message and proceed with 'while True:'
      print ("port is opened!")

    except IOError: # if port is already opened, close it and open it again and print message
      ser.close()
      ser.open()
      print ("port was already open, was closed and opened again!")
    XYZ_Location(11250,11250,8000,43430,45000,5000,ser)
    ser.close()
    
    app = Tk()

    q = Queue()
    r = Queue()

    myfont = font.Font(family='Arial',size=23,weight='bold')
    myfontlabel = font.Font(family='Arial',size=15)
    myfonttitle = font.Font(family='Arial',size=20,weight='bold')
    app.geometry('1668x1000')
    app.title('Drosophila Autoinjector GUI')
    
    Label(app, text='Robot', fg='black', font=myfonttitle,anchor='center').grid(row=0, column=0,sticky=W)
    fields = ['Dish number','Target pixel','Depth','Speed','Path']
    default = [1051,6000,-15,1000,'C:/Users/me-alegr011-admin/Downloads']
    values_1 = []
    for i in range(len(fields)):
       Label(app, text=fields[i], fg='black', font=myfontlabel).grid(row=i+1,column=0,sticky=W)
       e = Entry(app,fg='black',font=myfontlabel,width=18)
       e.insert(0, default[i])
       e.grid(row=i+1, column=1,sticky=W)
       values_1.append(e)     
    
    fields = ['Current Velocity X','Current Velocity Y','Current Velocity Z','Current X','Current Y','Current Z']
    default = [11250,11250,8000,57430,45000,10715]
    values_2 = []
    e_robot_1=[[],[],[],[],[],[]]
    for i in range(len(fields)):
       Label(app, text=fields[i], fg='black', font=myfontlabel).grid(row=i+7, column=0,sticky=W)
       e_robot_1[i] = Entry(app,fg='black',font=myfontlabel,width=18)
       e_robot_1[i].insert(0, default[i])
       e_robot_1[i].grid(row=i+7, column=1,sticky=W)
       values_2.append(e)
    
    Label(app, text='Robot stats', fg='black', font=myfonttitle,anchor='center').grid(row=13, column=0,sticky=W)
    fields = ['# Embryos detected','Current embryo','# Embryos injected','Need to change pipette?','Time for robot operation (min)','% of dish injected']
    default = [0,0,0,'No',0,0]
    values_s_1 = []
    e_stats_robot_1=[[],[],[],[],[],[]]
    for i in range(len(fields)):
       Label(app, text=fields[i], fg='black', font=myfontlabel).grid(row=i+14, column=0,sticky=W)
       e_stats_robot_1[i] = Entry(app,fg='black',font=myfontlabel,width=4)
       e_stats_robot_1[i].insert(0, default[i])
       e_stats_robot_1[i].grid(row=i+14, column=1,sticky=W)
       values_s_1.append(e_stats_robot_1[i])
    
    p1 = Process(target = stream_function_multi_process, args=(q,))
    p2 = Process(target = Main_code, args=(values_1[0].get(),values_1[1].get(),values_1[2].get(),values_1[3].get(),values_1[4].get(),q,r,))

    var=tk.IntVar()
    var_2=tk.IntVar()
    var_4=tk.IntVar()
    b1 = Button(app, text='Start robot', fg='red', width=11, font=myfont, command=lambda: [Main_code_process(p2,__name__),var.set(1)])
    b1.grid(row=6, column=0,columnspan = 1, sticky=W) 
    b2 = Button(app, text='Stop robot', fg='red', width=11, font=myfont, command=lambda: var_2.set(1))
    b2.grid(row=6, column=1,columnspan = 1, sticky=W)
    b3 = Button(app, text='Go to position', fg='red', width=11, font=myfont, command=lambda: go_to_position_new())
    b3.grid(row=13, column=1,columnspan = 1, sticky=W)
    b4 = Button(app, text='Start stream', fg='red', width=11, font=myfont, command=lambda: [Stream_code_process(p1,__name__),var_4.set(1)])
    b4.grid(row=13, column=0,columnspan = 1, sticky=W)

    img_1=Label(app)
    img_2=Label(app)
    dslr=Label(app)
    dslr_ml=Label(app)
    dslr.grid(row=0, column=3 ,rowspan=10,sticky=W) 
    dslr_ml.grid(row=10, column=3 ,rowspan=10,sticky=W) 
    img_1.grid(row=0, column=2 ,rowspan=10,sticky=W)
    img_2.grid(row=10, column=2 ,rowspan=10,sticky=W)
    app.update()
    
    print("waiting...")
    b4.wait_variable(var_4)
    print("done waiting.")
    time.sleep(10)
    while True and int(var_2.get())!=1:
        images= q.get(timeout=30)
        if images[0]==1:
            img=r.get()
            b,g,red=cv2.split(img[0])
            img = cv2.merge((red,g,b))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            dslr.configure(image=img)
            dslr.image = img
        elif images[0]==2:
            img=r.get()
            b,g,red=cv2.split(img[0])
            img = cv2.merge((red,g,b))
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            dslr_ml.configure(image=img)
            dslr_ml.image = img
        elif images[0]==3:
            vals=r.get()
            e_stats_robot_1[0].delete(0, 'end')
            e_stats_robot_1[0].insert(0, vals[0])
            e_stats_robot_1[1].delete(0, 'end')
            e_stats_robot_1[1].insert(0, vals[1])
            e_stats_robot_1[2].delete(0, 'end')
            e_stats_robot_1[2].insert(0, vals[2])
            e_stats_robot_1[3].delete(0, 'end')
            e_stats_robot_1[3].insert(0, vals[6])
            e_stats_robot_1[4].delete(0, 'end')
            e_stats_robot_1[4].insert(0, vals[7])
            e_stats_robot_1[5].delete(0, 'end')
            e_stats_robot_1[5].insert(0, vals[8])
            e_robot_1[3].delete(0, 'end')
            e_robot_1[3].insert(0, vals[3])
            e_robot_1[4].delete(0, 'end')
            e_robot_1[4].insert(0, vals[4])
            e_robot_1[5].delete(0, 'end')
            e_robot_1[5].insert(0, vals[5])
        elif len(images)==3:
            b,g,red=cv2.split(images[1])
            images1 = cv2.merge((red,g,b))
            b,g,red=cv2.split(images[2])
            images2 = cv2.merge((red,g,b))
            frame_1_rs = Image.fromarray(images1)
            frame_2_rs = Image.fromarray(images2)
            frame_1_rs = ImageTk.PhotoImage(frame_1_rs)
            frame_2_rs = ImageTk.PhotoImage(frame_2_rs)
            img_1.configure(image=frame_1_rs)
            img_2.configure(image=frame_2_rs)
            img_1.image = frame_1_rs
            img_2.image = frame_2_rs
        else:
            l=1
        app.update()
        

    p1.terminate()
    p2.terminate()
    app.mainloop()
