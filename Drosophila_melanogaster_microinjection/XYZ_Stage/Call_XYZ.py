import serial
import time
from XYZ_Position import *
from image_stich import *

XYZ_Location(1000,1000,1000,5500,3000,2500)
#XYZ_Location(5000,5000,10000,1000,1000,1000)
#XYZ_Positive(1000,1000,1000,0,850,0)
#XYZ_Negative(1000,1000,1000,1000,1000,0)
#XYZ_Location(5000,5000,10000,0,0,0)
#mer_imag = merge_images('C:/Users/asjos/Anaconda2/Projects/UMP Manipulator/Images/Trial1/starting_pos_needle_trial_1','C:/Users/asjos/Anaconda2/Projects/UMP Manipulator/Images/Trial1/starting_pos_needle_trial_2')

'''
mer_imag_x = Image.new('RGB',(0,0))
mer_imag_y = Image.new('RGB',(0,0))
m = 3
n = 4
for i in range (1, m+1):
    #print(i)
    mer_imag_x = Image.new('RGB', (0, 0))
    for j in range (1,n+1):
        #print(j)
        num = ((i-1)*n) + j
        print(num)
        image = Image.open('Images/starting_pos_needle_trial_{}.jpg'.format(num))
        mer_imag_x = merge_images_x(mer_imag_x,image)
        #print(mer_imag_x)
    mer_imag_y = merge_images_y(mer_imag_y,mer_imag_x)
    #print(mer_imag_y)

print (mer_imag_y)
mer_imag_y.save('Images/merged_image.jpg')
'''