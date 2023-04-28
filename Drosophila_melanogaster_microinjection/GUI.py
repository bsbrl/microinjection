from tkinter import *
import tkinter.messagebox
from PIL import ImageTk, Image
import serial
import time
from XYZ_Stage.XYZ_Position import *
from UMP_Manipulator.Camera_on import *
#from UMP_Manipulator.Path_GUI import *
from UMP_Manipulator.Camera import *
from UMP_Manipulator.Camera_on import *
#from UMP_Manipulator.Position import *
from UMP_Manipulator.edge_detection_needle5 import *
from UMP_Manipulator.sorted_path import *
from UMP_Manipulator.Injection import *
import numpy as np
import math
#from Z_axis_detection.z_axis_detection_XYZ import *

ser = serial.Serial\
(
    port='COM5',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)
app = Tk()

def Calibration():
    XYZ_Calibration()

def go_to_pos_GUI(values):
   XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()),
                int((values[4]).get()), int((values[5]).get()),ser)

def camera_on_off(number):
   tkMessageBox.showinfo("Camera","Press 'Y' to close camera")
   camera(number)

#def UMP_Call(speed, depth, X, Y):
#   UMP_Path(speed, depth, X, Y)

def focus_XYZ(values):
   auto_focus_XYZ(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()),
                  int((values[4]).get()), int((values[5]).get()),ser)
   
def move2x_up(values):
    XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get())+int((values[6]).get()), 
              int((values[4]).get()), int((values[5]).get()),ser)
    m = int((values[3]).get()) + int((values[6]).get())
    values[3].delete(0, END)
    values[3].insert(10, m)
    
def move2x_down(values):
    XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get())-int((values[6]).get()), 
              int((values[4]).get()), int((values[5]).get()),ser)
    m = int((values[3]).get()) - int((values[6]).get())
    values[3].delete(0, END)
    values[3].insert(10, m)
    
def move2x_left(values):
    XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()), 
              int((values[4]).get())-int((values[7]).get()), int((values[5]).get()),ser)
    m = int((values[4]).get()) - int((values[7]).get())
    values[4].delete(0, END)
    values[4].insert(10, m)
    
def move2x_right(values):
    XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()), 
              int((values[4]).get())+int((values[7]).get()), int((values[5]).get()),ser)
    m = int((values[4]).get()) + int((values[7]).get())
    values[4].delete(0, END)
    values[4].insert(10, m)
    
def move2x_z_up(values):
    XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()), 
              int((values[4]).get()), int((values[5]).get()) + int((values[8]).get()),ser)
    m = int((values[5]).get()) + int((values[8]).get())
    values[5].delete(0, END)
    values[5].insert(10, m)
    
def move2x_z_down(values):
    XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()), 
              int((values[4]).get()), int((values[5]).get()) - int((values[8]).get()),ser)
    m = int((values[5]).get()) - int((values[8]).get())
    values[5].delete(0, END)
    values[5].insert(10, m)
    
def change_2x4x(choice, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values):
    if choice == 1:
        default = default_d_10
    if choice == 2:
        default = default_d_50
    if choice == 3:
        default = default_d_100
    if choice == 4:
        default = default_d_500 
    if choice == 5:
        default = default_d_1000 
    values[6].delete(0, END)
    values[6].insert(10, default[0])
    values[7].delete(0, END)
    values[7].insert(10, default[1])
    values[8].delete(0, END)
    values[8].insert(10, default[2])
    
def move_center(default_center, values):
    values[3].delete(0, END)
    values[3].insert(10, default_center[0])
    values[4].delete(0, END)
    values[4].insert(10, default_center[1])
    values[5].delete(0, END)
    values[5].insert(10, default_center[2])
    XYZ_Location(int((values[0]).get()), int((values[1]).get()), int((values[2]).get()), int((values[3]).get()), 
              int((values[4]).get()), int((values[5]).get()),ser)

fields = 'Velocity X', 'Velocity Y', 'Velocity Z', 'Position X', 'Position Y', 'Position Z'
fields_d = 'dX', 'dY', 'dZ'
default = [112500, 112500, 8000, 57430, 45000, 10715]
default_center = [57943, 87788, 15653]
default_d_10 = [10, 10, 10]
default_d_50 = [50, 50, 50]
default_d_100 = [100, 100, 100]
default_d_500 = [500, 500, 500]
default_d_1000 = [1000, 1000, 1000]
speed = 4000
depth = 1500000
X = np.array([[896.169], [407.258], [805.444], [361.895], [850.806], [870.968], [951.613]])
Y = np.array([[923.387], [893.145], [656.250], [424.395], [137.097], [71.5726], [76.6129]])
num = len(fields)
num_d = len(fields_d)
values = []
number = 1

for i in range(0, num):
   Label(app, text=fields[i]).grid(row=i)
   e = Entry(app)
   e.insert(0, default[i])
   e.grid(row=i, column=1)
   values.append(e)

for j in range(0, num_d):
    Label(app, text=fields_d[j]).grid(row=j, column=3)
    z = Entry(app)
    z.insert(0, default_d_1000[j])
    z.grid(row = j, column=4)
    values.append(z)

img_up = ImageTk.PhotoImage(Image.open('GUI_Image/Up_arrow.png'))
img_down = ImageTk.PhotoImage(Image.open('GUI_Image/Down_arrow.png'))
img_left = ImageTk.PhotoImage(Image.open('GUI_Image/Left_arrow.png'))
img_right = ImageTk.PhotoImage(Image.open('GUI_Image/Right_arrow.png'))
img_center = ImageTk.PhotoImage(Image.open('GUI_Image/Center.png'))

b1 = Button(app, text='Quit', command=app.quit)
b1.grid(row=6, column=0, sticky=W, pady=4)
b2 = Button(app, text='Go to Position', command=lambda: go_to_pos_GUI(values))
b2.grid(row=6, column=1, sticky=W, pady=4)
b3 = Button(app, text='Camera On', command=lambda: camera_on_off(1))
b3.grid(row=7, column=0, sticky=W, pady=4)
#b4 = Button(app, text='UMP On', command=lambda: UMP_Call(speed, depth, X, Y))
#b4.grid(row=7, column=1, sticky=W, pady=4)
b5 = Button(app, text='Auto Focus', command=lambda: focus_XYZ(values))
b5.grid(row=8, column=0, sticky=W, pady=4)
b6 = Button(app, text='UP', image = img_up, command=lambda: move2x_up(values))
b6.grid(row=6, column=3, sticky=W, pady=4)
b7 = Button(app, text='DOWN', image = img_down, command=lambda: move2x_down(values))
b7.grid(row=8, column=3, sticky=W, pady=4)
b8 = Button(app, text='LEFT', image = img_left, command=lambda: move2x_left(values))
b8.grid(row=7, column=2, sticky=W, pady=4)
b9 = Button(app, text='RIGHT', image = img_right, command=lambda: move2x_right(values))
b9.grid(row=7, column=4, sticky=W, pady=4)
b10 = Button(app, text='_10_', anchor = CENTER, command=lambda: change_2x4x(1, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
b10.grid(row=0, column=5, sticky=W, pady=4)
b11 = Button(app, text='_50_', anchor = CENTER, command=lambda: change_2x4x(2, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
b11.grid(row=1, column=5, sticky=W, pady=4)
b12 = Button(app, text='_100', anchor = CENTER, command=lambda: change_2x4x(3, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
b12.grid(row=2, column=5, sticky=W, pady=4)
b13 = Button(app, text='_500', anchor = CENTER, command=lambda: change_2x4x(4, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
b13.grid(row=3, column=5, sticky=W, pady=4)
b14 = Button(app, text='1000', anchor = CENTER, command=lambda: change_2x4x(5, default_d_10, default_d_50, default_d_100, default_d_500, default_d_1000, values))
b14.grid(row=4, column=5, sticky=W, pady=4)
b15 = Button(app, text='Z_up', image = img_up, command=lambda: move2x_z_up(values))
b15.grid(row=6, column=5, sticky=W, pady=4)
b16 = Button(app, text='Z_down', image = img_down, command=lambda: move2x_z_down(values))
b16.grid(row=8, column=5, sticky=W, pady=4)
b17 = Button(app, text='__Z__', anchor = CENTER)
b17.grid(row=7, column=5, sticky=W, pady=4)
b18 = Button(app, text='Center', image = img_center, command=lambda: move_center(default_center, values))
b18.grid(row=7, column=3, sticky=W, pady=4)
b19 = Button(app, text='Cali', command=lambda: Calibration())
b19.grid(row=9, column=5, sticky=W, pady=4)

app.mainloop()
ser.close()