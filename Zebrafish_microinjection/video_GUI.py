# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:23:02 2020

@author: admin
"""


import cv2
from PIL import Image
from PIL import ImageTk
import threading
import tkinter as tk
from CameraWorkbench_master.camera import *
import time


def camera_on(videoloop_stop):
    threading.Thread(target=videoLoop, args=(videoloop_stop,)).start()


def camera_off(videoloop_stop):
    videoloop_stop[0] = True


def videoLoop(mirror=False):
    No = 0
    S=AmscopeCamera(0)
    S.activate()
    time.sleep(2)

    while True:
        frame=S.get_frame()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if mirror is True:
            frame = frame[:, ::-1]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel = tk.Label(image=image)
        panel.image = image
        panel.place(x=50, y=50)

        # check switcher value
        if videoloop_stop[0]:
            # if switcher tells to stop then we switch it again and stop videoloop
            videoloop_stop[0] = False
            panel.destroy()
            break


# videoloop_stop is a simple switcher between ON and OFF modes
videoloop_stop = [False]

root = tk.Tk()
root.geometry("1920x1080+0+0")

button1 = tk.Button(
    root, text="start", bg="#fff", font=("", 50),
    command=lambda: camera_on(videoloop_stop))
button1.place(x=1000, y=100, width=400, height=250)

button2 = tk.Button(
    root, text="stop", bg="#fff", font=("", 50),
    command=lambda: camera_off(videoloop_stop))
button2.place(x=1000, y=360, width=400, height=250)

root.mainloop()


'''
import sys
import cv2
import threading
import tkinter as tk
import tkinter.ttk as ttk
from queue import Queue
from PIL import Image
from PIL import ImageTk


class App(tk.Frame):
    def __init__(self, parent, title):
        tk.Frame.__init__(self, parent)
        self.is_running = False
        self.thread = None
        self.queue = Queue()
        self.photo = ImageTk.PhotoImage(Image.new("RGB", (800, 600), "white"))
        parent.wm_withdraw()
        parent.wm_title(title)
        self.create_ui()
        self.grid(sticky=tk.NSEW)
        self.bind('<<MessageGenerated>>', self.on_next_frame)
        parent.wm_protocol("WM_DELETE_WINDOW", self.on_destroy)
        parent.grid_rowconfigure(0, weight = 1)
        parent.grid_columnconfigure(0, weight = 1)
        parent.wm_deiconify()

    def create_ui(self):
        self.button_frame = ttk.Frame(self)
        self.stop_button = ttk.Button(self.button_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.RIGHT)
        self.start_button = ttk.Button(self.button_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.RIGHT)
        self.view = ttk.Label(self, image=self.photo)
        self.view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def on_destroy(self):
        self.stop()
        self.after(20)
        if self.thread is not None:
            self.thread.join(0.2)
        self.winfo_toplevel().destroy()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.is_running = False

    def videoLoop(self, mirror=False):
        No=0
        S=AmscopeCamera(0)
        S.activate()
        time.sleep(2)
        # cap = cv2.VideoCapture(No)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        while self.is_running:
            ret, to_draw = cap.read()
            if mirror is True:
                to_draw = to_draw[:,::-1]
            image = cv2.cvtColor(to_draw, cv2.COLOR_BGR2RGB)
            self.queue.put(image)
            self.event_generate('<<MessageGenerated>>')

    def on_next_frame(self, eventargs):
        if not self.queue.empty():
            image = self.queue.get()
            image = Image.fromarray(image)
            self.photo = ImageTk.PhotoImage(image)
            self.view.configure(image=self.photo)


def main(args):
    root = tk.Tk()
    app = App(root, "OpenCV Image Viewer")
    root.mainloop()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
'''