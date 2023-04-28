# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:46:35 2022

@author: User
"""

def stream_function_multi_process(q):
    import base64
    import cv2
    import zmq
    import numpy as np
    
    context = zmq.Context()
    footage_socket_1 = context.socket(zmq.PUB)
    footage_socket_1.setsockopt(zmq.LINGER, 0)
    footage_socket_1.connect('tcp://localhost:5555')
    footage_socket_2 = context.socket(zmq.PUB)
    footage_socket_2.setsockopt(zmq.LINGER, 0)
    footage_socket_2.connect('tcp://localhost:4555')
    footage_socket_3 = context.socket(zmq.SUB)
    try:
        footage_socket_3.bind('tcp://*:3555')
    except:
        print('socket already in use')
    footage_socket_3.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    cap_1=cv2.VideoCapture(1)
    cap_2=cv2.VideoCapture(0)
    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_1.set(cv2.CAP_PROP_FPS,30)
    cap_1.set(cv2.CAP_PROP_AUTOFOCUS,1)
    cap_2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap_2.set(cv2.CAP_PROP_FPS,30)
    cap_2.set(cv2.CAP_PROP_AUTOFOCUS,1)
    
    video_on=1
    inj_num=1
    while video_on==1:
        grabbed_1, frame_1 = cap_1.read()  # grab the current frame
        grabbed_2, frame_2 = cap_2.read()  # grab the current frame
        check=footage_socket_3.poll(1)
        if check==1:
            inj_num=footage_socket_3.recv()
            inj_num=int(inj_num.decode('utf-8'))
            encoded_1, buffer_1 = cv2.imencode('.jpg', frame_1)
            encoded_2, buffer_2 = cv2.imencode('.jpg', frame_2)
            jpg_as_text_1 = base64.b64encode(buffer_1)
            jpg_as_text_2 = base64.b64encode(buffer_2)
            rec=0
            while rec!=-1:
                footage_socket_1.send(jpg_as_text_1)
                footage_socket_2.send(jpg_as_text_2)
                check=footage_socket_3.poll(1)
                if check==1:
                    num=footage_socket_3.recv()
                    rec=int(num.decode('utf-8'))
        else:
            inj_num=inj_num
        inj_num=str(inj_num)
        cv2.putText(frame_1, 'Embryo '+inj_num, (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,cv2.LINE_AA)
        cv2.putText(frame_2, 'Embryo '+inj_num, (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2,cv2.LINE_AA)
        frame_1 = cv2.resize(frame_1, (889, 500))  # resize the frame
        frame_2 = cv2.resize(frame_2, (889, 500))  # resize the frame
        if q.empty()==True:
            q.put([0,frame_1,frame_2])
    cap_1.release()
    cap_2.release()
    footage_socket_1.close()
    footage_socket_2.close()
    footage_socket_3.close()
            
def Main_code(dish_num,target_pixel,inj_depth,inj_speed,path,q,r):
    # import os
    from XYZ_Stage.XYZ_Position import XYZ_Location
    from DSLR_Camera.DSLR_Call import func_TakeNikonPicture
    from ML.ml_whole_image import ml
    from ML.order import order 
    import time
    import numpy as np
    from injection_ml_tip_short_new_thresh import injection_ml_tip_short_new_thresh
    from new_pipette_new import new_pipette_new
    from first_pipette import first_pipette
    from path_finder_new import path_finder_new
    from ML.detections_dslr_image import detections_dslr_image
    import tensorflow as tf
    import math
    import serial
    from ML.transformation_matrix_DSLR_pipette import function_transformation_matrix_DSLR_pipette
    from new_z import new_z
    from injection_results_new import injection_results_new
    import cv2
    import zmq
    
    # os.system("taskkill /im python.exe /F")

    total_start_time=time.time()
    dish_num=str(dish_num)
    target_pixel=int(target_pixel)
    inj_depth=int(inj_depth)
    inj_speed=int(inj_speed)
    # Initial Variables
    # small dish height and light should be 2 intensity, 6 for regular
    z_needle=16000
    # z_needle=19000
    Z_initial=z_needle+5000      
    width_image=6000
    height_image=4000
    thresh_ml=.1
    sum_image_thresh_max=20000
    # sum_image_thresh_min=3000
    sum_image_thresh_min=1500
    over_injected=0
    missed=0
    # inj_num=int(input('inj_num = '))
    inj_num=0
    over_injected=0
    missed=0
    no_injected=0
    inj_num_init=inj_num
    pressure_value=1
    back_pressure_value=10
    pressure_time=4
    # inj_depth=-15
    post_z=-200
    pipette=1
    calib_pipette_num=1
    pip_num=0
    pip_em_num=[0]
    switch_list=[]
    injected_embryos=0
    injected_embryos_count=0
    injected=2
    miss=1
    injection_list=[]
    injected_list=[]
    elim_embryo=[]
    deltas_pipette=[[0,0,0]]
    injection_time_list=[]
    inj_time_total=0
    # Open sockets
    context = zmq.Context()
    footage_socket_1 = context.socket(zmq.SUB)
    try:
        footage_socket_1.bind('tcp://*:5555')
    except:
        print('socket already in use')
    footage_socket_1.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
    footage_socket_2 = context.socket(zmq.SUB)
    try:
        footage_socket_2.bind('tcp://*:4555')
    except:
        print('socket already in use')
    footage_socket_2.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
    footage_socket_3 = context.socket(zmq.PUB)
    footage_socket_3.connect('tcp://localhost:3555')
    time.sleep(.1)
    
    # Connect XYZ stage
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


    # Connect to arduino
    try:
      arduino = serial.Serial(
        port="COM9",
        baudrate=9600,
      )
      arduino.isOpen() # try to open port, if possible print message and proceed with 'while True:'
      print ("port is opened!")
    
    except IOError: # if port is already opened, close it and open it again and print message
      arduino.close()
      arduino.open()
      print ("port was already open, was closed and opened again!")
    
    time.sleep(5)
    print('Connecting to arduino')

    # Go to camera 
    print('Moving under DSLR')
    # take picture
    filename='Entire_Petri_Dish_'+dish_num+'.jpg'
    XYZ_Location(10000,10000,8000,54430,127000,5000,ser)
    q.put([3])
    r.put([0,0,0,54430,127000,5000,'No',0,0])
    np.save('currentXY.npy',np.array([54430,127000]))
    time.sleep(20)
    func_TakeNikonPicture(filename)
    time.sleep(10)
    image=cv2.imread(path+'/Robot_code/DSLR_Camera/'+filename)
    image=image[450:3485,510:3545]
    img_dish_gui=cv2.resize(image,(500,500))
    cv2.imwrite(path+'/Robot_code/DSLR_Camera/'+'gui_'+filename,img_dish_gui)
    q.put([1])
    r.put([img_dish_gui])

    # detect embryos
    print('Detecting embryos')
    xc_rc,yc_rc,scores=ml(path+'/Robot_code/faster_r_cnn_trained_model_petri_new_8',path+'/Robot_code/DSLR_Camera/'+filename,thresh_ml,width_image,height_image,tf,np)
    img_dish=cv2.imread(path+'/Robot_code/DSLR_Camera/'+filename,1)
    y1a_rc,y2a_rc,x1a_rc,x2a_rc=order(xc_rc,yc_rc,0,0)

    for i in range(len(y1a_rc)):
        # cv2.rectangle(img_dish_new,(int(x1a_rc[i][0]*(float(1600)/float(6000))),int(y1a_rc[i][0]*(float(1067)/float(4000)))),(int(x2a_rc[i][0]*(float(1600)/float(6000))),int(y2a_rc[i][0]*(float(1067)/float(4000)))),(0,255,0),1)
        # cv2.rectangle(img_dish_gui,(int(x1a_rc[i][0]*(float(610)/float(6000))),int(y1a_rc[i][0]*(float(407)/float(4000)))),(int(x2a_rc[i][0]*(float(610)/float(6000))),int(y2a_rc[i][0]*(float(407)/float(4000)))),(0,255,0),1)
        cv2.rectangle(img_dish,(int(x1a_rc[i][0]),int(y1a_rc[i][0])),(int(x2a_rc[i][0]),int(y2a_rc[i][0])),(0,255,0),3)
    img_dish=img_dish[450:3485,510:3545]
    img_dish_gui=cv2.resize(img_dish,(500,500))

    xc_rc_new_list=[]
    yc_rc_new_list=[]
    x1a_rc_new_list=[]
    y1a_rc_new_list=[]
    x2a_rc_new_list=[]
    y2a_rc_new_list=[]
    for i in range(len(xc_rc)):
        xc_rc_new_list.append(int(xc_rc[i][0]))
        yc_rc_new_list.append(int(yc_rc[i][0]))
        x1a_rc_new_list.append(int(xc_rc[i][1]))
        y1a_rc_new_list.append(int(yc_rc[i][1]))
        x2a_rc_new_list.append(int(xc_rc[i][2]))
        y2a_rc_new_list.append(int(yc_rc[i][2]))
    cv2.imwrite(path+'/Robot_code/DSLR_Camera/ML_Petri_Dishes/'+filename,img_dish)
    cv2.imwrite(path+'/Robot_code/Video images/ML_image/'+filename,img_dish)
    
    cv2.imwrite(path+'/Robot_code/DSLR_Camera/ML_Petri_Dishes/'+'gui_'+filename,img_dish_gui)
    q.put([2])
    r.put([img_dish_gui])
    print('Finished detecting embryos')
    print('Number of embryos = {}'.format(len(xc_rc)))
    time.sleep(5)
    q.put([0])

    # Open new tensorflow session
    graph = tf.Graph()
    with graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v2.io.gfile.GFile(path+'/Robot_code/faster_r_cnn_trained_model_injection_point_tip_pipette_robot_2_new_4'+'/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            img_dish=cv2.imread(path+'/Robot_code/DSLR_Camera/'+filename,1)
            x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post,xc_rc_keep,yc_rc_keep=path_finder_new(xc_rc_new_list,yc_rc_new_list,x1a_rc_new_list,y1a_rc_new_list,x2a_rc_new_list,y2a_rc_new_list,img_dish,filename)
            img_path=cv2.imread(path+'/Robot_code/Video images/Path_image/'+filename)
            img_path=img_path[450:3485,510:3545]
            img_path_gui=cv2.resize(img_path,(500,500))
            cv2.imwrite(path+'/Robot_code/Video images/Path_image/'+'gui_'+filename,img_path_gui)
            q.put([1])
            r.put([img_path_gui])
            
            positions=[]
            for i in range(len(xc_rc_keep)):
                embryo_point_center = function_transformation_matrix_DSLR_pipette(xc_rc_keep[i],yc_rc_keep[i],1529,1464,2661,1826,1810,2427,54230,52800,33030,45900,49070,34650) # DSLR to pipette
                positions.append([int(float(embryo_point_center.item(0,0))),int(float(embryo_point_center.item(1,0)))])
            print(len(positions))
            
            for pic in range(len(positions)):
                total_start_time_inj=time.time()
                print('Embryo {} out of {} Embryos'.format(pic+1,len(positions)))
                cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (0, 0, 0), thickness=3)
                img_dish_small=img_dish[450:3485,510:3545]
                img_path_gui=cv2.resize(img_dish_small,(500,500))
                q.put([1])
                r.put([img_path_gui])
                if pic==0:
                    print('New X = ',positions[pic][0])
                    print('New Y = ',positions[pic][1])
                    print('New Z = ',z_needle-300)
                    dx_final=0
                    dy_final=0
                    dz=0
                    view_1_x=560
                    view_1_y=161
                    view_2_x=606
                    view_2_y=339
                    time_wait=4.1
                    print('Start air pressure')
                    print('Start stream')
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    np.save('currentXY.npy',np.array([positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final)]))
                    footage_socket_1,footage_socket_2,z_needle_new,dx_final,dy_final,view_1_x,view_1_y,view_2_x,view_2_y=first_pipette(view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,Z_initial,pic,arduino)
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    np.save('currentXY.npy',np.array([positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final)]))
                    dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value_current,injection_time,miss=injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle_new,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel,miss)
                else:
                    print('New X = ',positions[pic][0])
                    print('New Y = ',positions[pic][1])
                    print('New Z = ',Z_new )
                    dist=math.hypot(positions[pic-1][0]-positions[pic][0],positions[pic-1][1]-positions[pic][1])
                    cv2.line(img_dish, (xc_rc_keep[pic-1], yc_rc_keep[pic-1]), (xc_rc_keep[pic], yc_rc_keep[pic]), (0, 125, 0), thickness=3)
                    print('Distance traveled = ',dist)
                    if dist>12000:
                        Z_new=new_z(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,pip_num,Z_inj_actual,pic)
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),Z_new,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    np.save('currentXY.npy',np.array([positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final)]))
                    dx_final,dy_final,X_inj,Y_inj,Z_inj,footage_socket_1,footage_socket_2,injection_list_num,Z_new,dz,view_1_x,view_1_y,view_2_x,view_2_y,Z_inj_actual,pipette,current_x_centroid,current_y_centroid,current_z_centroid,injected,sum_image,pressure_value_current,injection_time,miss=injection_ml_tip_short_new_thresh(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),Z_new,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,pic,sum_image_thresh_min,target_pixel,miss)
                inj_num+=1
                pipette=0
                
                if injection_time!=0:
                    injection_time_list.append(injection_time)
                if -10<int(sum_image)<sum_image_thresh_min:
                    switch_list.append(1)
                elif int(sum_image)>=sum_image_thresh_min:
                    switch_list.append(0)
                else:
                    print('append nothing')
                if injected==2:
                    missed+=1
                    print('Missed injection')
                    elim_embryo.append([current_x_centroid,current_y_centroid,current_z_centroid,pip_num,pic])
                    print('Number of injected embryos = ',injected_embryos)
                    print('Remove embryo from dish')
                    print('Number of embryos missed = ',missed)
                    injection_list.append(4) 
                    cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (0,0,255), thickness=3)
                elif int(sum_image)<sum_image_thresh_min:
                    no_injected+=1
                    print('No injection')
                    elim_embryo.append([current_x_centroid,current_y_centroid,current_z_centroid,pip_num,pic])
                    print('Number of injected embryos = ',injected_embryos)
                    print('Remove embryo from dish')
                    print('Number of embryos not injected = ',no_injected)
                    injection_list.append(4) 
                    injected_list.append(0) 
                    cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (0,0,255), thickness=3)
                else:
                    if sum_image>sum_image_thresh_max and injected!=2 and int(sum_image)!=0:
                        over_injected+=1
                        pressure_time=4
                        injected_embryos+=1
                        injected_embryos_count+=1
                        print('Number of injected embryos = ',injected_embryos)
                        print('Remove embryo from dish')
                        print('Number of embryos over injected = ',over_injected)
                        injection_list.append(4) 
                        injected_list.append(1) 
                        cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (171,118,55), thickness=3)
                    elif sum_image_thresh_min<sum_image<sum_image_thresh_max and injected!=2 and int(sum_image)!=0:
                        injection_list.append(injection_list_num)
                        print('Successful injection')
                        injected_embryos+=1
                        injected_embryos_count+=1
                        print('Number of injected embryos = ',injected_embryos)
                        injected_list.append(1)
                        cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (171,118,55), thickness=3)
                    else:
                        print('Good')
                        cv2.rectangle(img_dish, (xc_rc_keep[pic]-20, yc_rc_keep[pic]-20), (xc_rc_keep[pic]+20, yc_rc_keep[pic]+20), (171,118,55), thickness=3)
                total_end_time_inj=time.time()
                inj_time=int(total_end_time_inj-total_start_time_inj)
                inj_time_total+=inj_time
                img_dish_small=img_dish[450:3485,510:3545]
                img_path_gui=cv2.resize(img_dish_small,(500,500))
                q.put([1])
                r.put([img_path_gui])
                q.put([3])
                r.put([len(positions),pic+1,injected_embryos,current_x_centroid,current_y_centroid,current_z_centroid,'No',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                np.save('currentXY.npy',np.array([positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final)]))
                
                if len(switch_list)>2 and switch_list[len(switch_list)-3]==1 and switch_list[len(switch_list)-2]==1 and switch_list[len(switch_list)-1]==1:
                    print('CHANGE TO NEW PIPETTE AND VALVES!')
                    q.put([3])
                    r.put([len(positions),pic+1,injected_embryos,current_x_centroid,current_y_centroid,current_z_centroid,'Yes',int(inj_time_total/60),int(100*(injected_embryos/len(positions)))])
                    np.save('currentXY.npy',np.array([positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final)]))
                    miss=1
                    pressure_value=1
                    pressure_time=4
                    time_wait=10
                    injected_list=[]
                    switch_list=[]
                    pipette=1
                    pip_num+=1
                    calib_pipette_num+=1
                    pip_em_num.append(pic+1)
                    dx_final,dy_final,current_x,current_y,current_z,footage_socket_1,footage_socket_2,Z_new,view_1_x,view_1_y,view_2_x,view_2_y,injected_embryos_count,dz_final,current_z_needle=new_pipette_new(time_wait,view_1_x,view_1_y,view_2_x,view_2_y,positions[pic][0]+int(dx_final),positions[pic][1]+int(dy_final),z_needle,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,ser,pip_num,Z_inj_actual,inj_depth,inj_speed,back_pressure_value,pressure_value,pressure_time,arduino,post_z,Z_initial,current_z_centroid,pic,sum_image_thresh_min,target_pixel) 
                    deltas_pipette.insert(pip_num-1,[current_x-current_x_centroid,current_y-current_y_centroid,current_z_needle-current_z_centroid])
                    if pic+1<len(positions):
                        XYZ_Location(20000,20000,8000,positions[pic+1][0]+int(dx_final),positions[pic+1][1]+int(dy_final),0,ser)
                        time.sleep(5)
                        XYZ_Location(20000,20000,8000,positions[pic+1][0]+int(dx_final),positions[pic+1][1]+int(dy_final),Z_new,ser)
                        time.sleep(5)
                print('Starting pressure = ',pressure_value)
            # Stop pressure
            time.sleep(1)
            arduino.write("P0p".encode())
            time.sleep(1)
            if injected_embryos==len(positions):
                XYZ_Location(20000,20000,8000,current_x,current_y,0,ser)
            else:
                if len(deltas_pipette)<2:
                    print('no adding deltas')
                else:
                    deltas_pipette_x=[]
                    deltas_pipette_y=[]
                    deltas_pipette_z=[]
                    for w in range(len(deltas_pipette)):
                        deltas_pipette_x.append(deltas_pipette[w][0])
                        deltas_pipette_y.append(deltas_pipette[w][1])
                        deltas_pipette_z.append(deltas_pipette[w][2])
                    for h in range(len(deltas_pipette)):
                        deltas_pipette[h]=[sum(deltas_pipette_x[h:len(deltas_pipette_x)]),sum(deltas_pipette_y[h:len(deltas_pipette_y)]),sum(deltas_pipette_z[h:len(deltas_pipette_z)])]
                elim_embryo_new=[]
                for q in range(len(elim_embryo)):
                    elim_embryo_new.append([elim_embryo[q][0]+deltas_pipette[elim_embryo[q][3]][0],elim_embryo[q][1]+deltas_pipette[elim_embryo[q][3]][1],elim_embryo[q][2]+deltas_pipette[elim_embryo[q][3]][2],elim_embryo[q][3],elim_embryo[q][4]])
                injection_results_new(elim_embryo_new,filename,view_1_x,view_1_y,view_2_x,view_2_y,dx_final,dy_final,footage_socket_1,footage_socket_2,footage_socket_3,inj_num,graph,sess,arduino,back_pressure_value,pressure_value,pressure_time,dz,inj_speed,inj_depth,inj_num_init,ser,pipette,post_z,current_x_centroid,current_y_centroid,current_z_centroid,len(positions))
    print('Press y on video stream')
    # Close sockets
    footage_socket_1.close()
    footage_socket_2.close()
    footage_socket_3.close()
    # Disconnect xyz stage
    ser.close()
    # Save petri dish and requisite injections  
    mypath=path+'/Robot_code/DSLR_Camera/Row_Col_Petri_Dish'
    path=path+'/Robot_code/DSLR_Camera/'        
    detections_dslr_image(path,filename,mypath,injection_list,x1a_rc_post,y1a_rc_post,x2a_rc_post,y2a_rc_post)
    total_end_time=time.time()
    print('Number of injected embryos = ',injected_embryos)
    print('{} % of dish injected'.format(float(injected_embryos)/float(len(positions))*100))
    print('Average time for injection (s) = ',np.mean(injection_time_list))
    print('Time for injection of dish (min) = ',int((total_end_time-total_start_time)/60)) 
    print('Injection pressure (psi) = ',pressure_value)
    print('Injection pressure time (s) = ',pressure_time)
    print('Injection depth (um) = ',inj_depth)
    print('Injection speed (um/s) = ',inj_speed)

# def Main_code_process(p1,p2,__name__act):
#     __name__=__name__act
#     if __name__=='__main__':
        
#         p1.start()
#         p2.start()
def Main_code_process(p2,__name__act):
    __name__=__name__act
    if __name__=='__main__':
        
        p2.start()
def Stream_code_process(p1,__name__act):
    __name__=__name__act
    if __name__=='__main__':
        
        p1.start()

