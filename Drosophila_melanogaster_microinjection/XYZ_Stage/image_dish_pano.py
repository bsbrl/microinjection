import serial
import time
from XYZ_Position import *
from image_stich_pano import *
from Camera import *

pics = []
#merged_img = cv2.imread('Images/starting_pos_needle_trial_5.jpg')
#'''
for i in range (1,14):
    print(i)
    #left = merged_img
    right = cv2.imread('Images/merged_overall_{}.png'.format(i))
    #merged_img = merge_images_pano_x(left,right)
    #time.sleep(0.5)
    pics.append(right)

stitcher = cv2.createStitcher(False)
merged_img = stitcher.stitch(pics)
cv2.imwrite('Images/merged_overall.png',merged_img[1])
#'''