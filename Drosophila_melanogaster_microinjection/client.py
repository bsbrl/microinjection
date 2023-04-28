# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:30:51 2022

@author: me-alegr011-admin
"""

import socket

def client(host,info,port):
    s=socket.socket()
    s.connect((host,port))
    fileToSend = open(info,'r')
    content = fileToSend.read()
    s.send(content.encode())

port=12000 #same as server
host='10.128.105.7'
info='C:/Users/me-alegr011-admin/Downloads/connection.txt'
client(host,info,port)