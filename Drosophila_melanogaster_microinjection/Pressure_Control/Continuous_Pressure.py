# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:54:50 2020

@author: enet-joshi317-admin
"""

def continuous_pressure(bp,injp,type_pressure):
    if type_pressure=='bp':
        value=int(bp)
        if value <=5:
            signal = int((value + 0.35)/0.107)
            signal = str(signal)
            P = "P"
            p = "p"
            signal = P + signal + p
            print ("Done with Backpressure", value, "PSI pressure.")
    
        else:
            signal = int((value + 43.6279)/0.9535)
            signal = str(signal)
            P = "P"
            p = "p"
            signal = P + signal + p
            print ("Done with Backpressure", value, "PSI pressure.")
    else:
        value=int(injp)
        if value <=5:
            signal = int((value + 0.35)/0.107)
            signal = str(signal)
            P = "P"
            p = "p"
            signal = P + signal + p
            print ("Done with", value, "PSI pressure.")
    
        else:
            signal = int((value + 43.6279)/0.9535)
            signal = str(signal)
            P = "P"
            p = "p"
            signal = P + signal + p
            print ("Done with", value, "PSI pressure.")
    return signal
# # s=continuous_pressure(0,20,'inj')
# import serial
# import time
# # from stream_function_multi_process import stream_function_multi_process
# # from multiprocessing import Queue, Process
# arduino = serial.Serial('COM9', 9600, timeout = 5)
# time.sleep(5)
# signal=continuous_pressure(0,30,'inj')
# arduino.write(signal.encode())
# # arduino.write("P0p".encode())
# # if __name__=='__main__':
# #     print('Starting')
# #     q=Queue()
# #     r=Queue()
# #     p=Process(target=stream_function_multi_process, args=(q,r,))
# #     # p=Process(target=stream_function_multi_process, args=(q,))
# #     p.start()
    
# #     arduino = serial.Serial('COM9', 9600, timeout = 5)
# #     time.sleep(5)
# #     back_pressure_value_new=0
# #     pressure_value=1
# #     injected=2
# #     count=0
# #     while pressure_value<=40 and injected==2:
# #         count+=1
# #         correct=0
# #         o=0
# #         while correct==0 and injected==2:
# #             print('Try ',o+1)
# #             signal=continuous_pressure(back_pressure_value_new,pressure_value,'inj')
# #             arduino.write(signal.encode())
# #             arduino.flush()
# #             q_=arduino.readline()
# #             q_=q_.decode()
# #             s=q_.find('Received')
# #             r.put([1])
# #             images=q.get()
                    
# #             if pressure_value>5:
# #                 press_num=int((pressure_value + 43.6279)/0.9535)
# #             else:
# #                 press_num=int((pressure_value + 0.35)/0.107)
# #             if pressure_value>5:
# #                 if press_num>99:
# #                     if q_[s+9]=='P' and q_[s+10:s+13]==str(int((pressure_value + 43.6279)/0.9535)) and q_[s+13]=='p' and q_[s+14]=='\r':
# #                         correct=1
# #                     else:
# #                         o+=1
# #                 else:
# #                     if q_[s+9]=='P' and q_[s+10:s+12]==str(int((pressure_value + 43.6279)/0.9535)) and q_[s+12]=='p' and q_[s+13]=='\r':
# #                         correct=1
# #                     else:
# #                         o+=1 
# #             else:
# #                 if press_num>9:
# #                     if q_[s+9]=='P' and q_[s+10:s+12]==str(int((pressure_value + 0.35)/0.107)) and q_[s+12]=='p' and q_[s+13]=='\r':
# #                         correct=1
# #                     else:
# #                         o+=1 
# #                 else:
# #                     if q_[s+9]=='P' and q_[s+10:s+11]==str(int((pressure_value + 0.35)/0.107)) and q_[s+11]=='p' and q_[s+12]=='\r':
# #                         correct=1
# #                     else:
# #                         o+=1 
# #         pressure_value+=1
# #     r.put([2])