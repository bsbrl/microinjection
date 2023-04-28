import cv2
import numpy as np
import time

def merge_images_pano_x(top,bottom):
    stitcher = cv2.createStitcher(False)
    pic1 = top
    pic2 = bottom
    time.sleep(2)
    result = stitcher.stitch((pic1,pic2))
    return result[1]
    
''' 
pic = cv2.imread('Images/Trial2/starting_pos_needle_trial_{}'.format(i))

stitcher = cv2.createStitcher(False)
result = stitcher.stitch((pic))

stitcher = cv2.createStitcher(False)
pic1 = cv2.imread('Images/starting_pos_needle_trial_5.jpg')
pic2 = cv2.imread('Images/starting_pos_needle_trial_6.jpg')
pic3 = cv2.imread('3.right_2.jpg')
result = stitcher.stitch((pic1,pic2))

cv2.imwrite('4.merged_image.jpg',result[1])
'''