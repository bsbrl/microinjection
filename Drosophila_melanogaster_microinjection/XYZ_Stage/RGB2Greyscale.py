from PIL import Image
import cv2
import numpy as np

img = Image.open('starting_pos_needle_trial_11.jpg').convert('LA')
img.save('greyscale.png')
pixel = cv2.imread('greyscale.png')
print(pixel[1079,1919,0])
#print(pixel[0,0])
for i in range(0, 1080):
    for j in range(0, 1920):
        if pixel[i,j,0] >= 75 and pixel[i,j,1] >= 75 and pixel[i,j,2] >= 75:
            pixel[i,j] = [255,0,0]

new_image = Image.fromarray(pixel)
new_image.save('new_image.png')