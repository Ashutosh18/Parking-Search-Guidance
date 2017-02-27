import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('new_fill.jpg')
img3 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img2 = cv2.imread('new_parking.jpg')
img4 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

img_subtract = img3 - img4
img5 = cv.cvtColor(img_subtract, cv.COLOR_GRAY2BGR)

kernel = np.ones((5,5),np.uint8)
img8 = cv.erode(img5, kernel, iterations =5)

ret, thresh1 = cv.threshold(img8, 127, 255, 0)
titles = ['Original Image', 'BINARY']
images = [img8, thresh1]
img6 = cv.cvtColor(thresh1, cv.COLOR_BGR2GRAY)
	
contours, hierarchy = cv.findContours(img6,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

if len(contours)>0:
    for i in range(0,len(contours),1):
        cnt= contours[i]
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        print cx
        cy = int(M['m01']/M['m00'])
        print cy
        dis = math.sqrt((cx - 350)**2 + (cy - 459)**2)
        print dis
        print('  ')
else:
print "sorry no lot is empty"
