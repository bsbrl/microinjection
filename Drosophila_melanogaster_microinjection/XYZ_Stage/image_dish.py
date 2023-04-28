import serial
import time
from XYZ_Position import *
from Camera import *
import cv2
import numpy as np
from image_stich_pano import *

XYZ_Location(1000,1000,1000,8500,2100,1500)

z = 1500
y1 = 2100
y2 = 9300
x1 = 8500
x2 = 1250
dy = 400
dx = 250
Cx = 4875
Cy = 5700
r = 3600
pic_x = []

m = (x1 - x2)/dx
n = (y2 - y1)/dy

#mer_imag_y = Image.new('RGB',(0,0))

for i in range (0, m+1):
    #print(i)
    XYZ_Location(1000,1000,1000, x1 - (i*dx), y1, z)
    time.sleep(1)
    #mer_imag_x = Image.new('RGB', (0, 0))
    pic_y = []
    img_num = 0
    for j in range (0,n+1):
        #print(j)
        num = ((i)*(n+1)) + (j+1)
        print(num)
        p1 = ((x1 - (i * dx)) - Cx) ** 2 + ((y1 + (j * dy)) - Cy) ** 2
        p2 = ((x1 - (i * dx)) - Cx) ** 2 + ((y1 + ((j+1) * dy)) - Cy) ** 2
        p3 = ((x1 - ((i+1) * dx)) - Cx) ** 2 + ((y1 + (j * dy)) - Cy) ** 2
        p4 = ((x1 - ((i+1) * dx)) - Cx) ** 2 + ((y1 + ((j+1) * dy)) - Cy) ** 2
        if (p1 < (r**2) and p2 < (r**2)) or (p1 < (r**2) and p3 < (r**2)) or (p1 < (r**2) and p4 < (r**2)) or (p2 < (r**2) and p3 < (r**2)) or (p2 < (r**2) and p4 < (r**2)) or (p3 < (r**2) and p4 < (r**2)):
            XYZ_Location(1000,1000,1000, x1 - (i*dx), y1 + (j*dy), z)
            image(num)
            #img_num = img_num +1
            temp_image_y = cv2.imread('Images/starting_pos_needle_trial_{}.jpg'.format(num))
            pic_y.append(temp_image_y)
            '''
            if img_num == 1:
                cv2.imwrite('Images/temp_image_y.png',temp_image)
            else:
                left = cv2.imread('Images/temp_image_y.png')
                right = cv2.imread('Images/starting_pos_needle_trial_{}.jpg'.format(num))
                merged_img = merge_images_pano_x(left,right)
                cv2.imwrite('Images/temp_image_y.png', merged_img)
            '''
        #mer_imag_x = merge_images_x(mer_imag_x,temp_image)
        #print(mer_imag_x)
    stitcher = cv2.createStitcher(False)
    merged_img = stitcher.stitch(pic_y)
    cv2.imwrite('Images/merged_image_y{}.png'.format(i+1), merged_img[1])
    temp_image_x = cv2.imread('Images/merged_image_y{}.png'.format(i+1))
    pic_x.append(temp_image_x)
    '''
    if i == 0:
        cv2.imwrite('Images/temp_image_x.png', merged_img[1])
    else:
        top = cv2.imread('Images/temp_image_x.png')
        bottom = cv2.imread('Images/merged_image_y{}.png'.format(i+1))
        merged_overall = merge_images_pano_x(top,bottom)
        cv2.imwrite('Images/temp_image_x.png',merged_overall)
    #XYZ_Negative(1000, 1000, 1000, 0, dy*(n+1), 0)
    #mer_imag_y = merge_images_y(mer_imag_y,mer_imag_x)
    #print(mer_imag_y)
    '''

stitcher = cv2.createStitcher(False)
merged_overall = stitcher.stitch(pic_x)
cv2.imwrite('Images/merged_overall.png', merged_overall[1])
#print (mer_imag_y)
#mer_imag_y.save('Images/merged_image.jpg')