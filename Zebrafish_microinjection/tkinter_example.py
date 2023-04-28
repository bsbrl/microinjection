# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 00:06:39 2020

@author: admin
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:27:51 2019
Example use of tkinter with python threading.
@author: Benedict Wilkins AI
"""
import tkinter as tk
from threading import Thread 
import time

class Sleep:

    def __init__(self, wait):
        self.wait = wait
       
    def __enter__(self):
        self.start = self.__t()
        self.finish = self.start + self.wait
    
    def __exit__(self, type, value, traceback):
        while self.__t() < self.finish:
            time.sleep(1./1000.)

    def __t(self):
        return int(round(time.time() * 1000))
            
def after(t, fun, *args):
    global finish
    if not finish:
        root.after(t, fun, *args)
    
def run():
    global finish
    x = -200
    while not finish:
        with Sleep(100):
            after(0, lambda : canvas.delete('all'))
            after(0, lambda : canvas.create_rectangle(x,0,x+200,200, fill='red'))
        x += 10
        if x > 400:
            x = -200

def quit():
    global finish
    finish = True
    root.destroy()

root = tk.Tk()

root.title("Test")
root.protocol("WM_DELETE_WINDOW", quit)

canvas = tk.Canvas(root, width=400, height=200, bd=0,
                   highlightthickness=0, bg='white')
canvas.pack()

global finish
finish = False

control_thread = Thread(target=run, daemon=True)
control_thread.start()

root.mainloop()
control_thread.join()