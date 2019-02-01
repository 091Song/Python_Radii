# -*- coding: utf-8 -*-
"""
Practice: OpenCV
    Basic Image processing: Read, show, write, convert color
    Extract image data
    
@author: Y.Song
"""
# about data type: rfirend.tistory.com/tag/dtype

### import 
# OpenCV 
import cv2
# Matplotlib
import matplotlib.pyplot as plt
# for calculation
import numpy as np 
# for data processing
import pandas as pd

### Read an image
# image file name
IFname = 'SampleImage.jpg'

# Read
#imgCL = cv2.imread('SampleImage.jpg', cv2.IMREAD_COLOR) # Color image
imgBW = cv2.imread('SampleImage.jpg', 0) # Black and White image
# 0 for black / 255 for white

### Show images
#cv2.imshow('Color', imgCL)
#cv2.imshow('BlackWhite', imgBW)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
## note: a keyboard input (such as click [enter or 0]) at an image window 
##      is necessary to proceed running other commands

### save an image
#cv2.imwrite('SampleGray.jpg', imgBW)

### 
# (h, w): height and widht of an image
h, w = imgBW.shape 

# set an array for a horizontal axis 
X = np.arange(w)
Y = np.arange(h)

# for new array: same data type as in the image
Intf = np.zeros((h,w), dtype = imgBW.dtype )
Intb = np.zeros(h)

for i in range(0, h):
    # an index for the darkest point at a hight i
    lmin = pd.Series(imgBW[i,:]).idxmin()
    # currently set white for a boundary value
    Intb[i] = h - lmin
    Intf[i,lmin] = 255 

########
# update later
    # if there are two data points which have same minimums
    # ignore the noisy points
########
        


# get the index of the minimum
# pd.Series(imgBW[int(h/2),:]).idxmin()
    
### This is for check image 
# to save the original image, copy it
imgProc = np.array(imgBW)

imgProc[h/2-10:h/2+10, :] = 255

### sort
#imgBW2 = (imgBW < 120 )

#matrix = cv2.getRotationMatrix2D((height/2,width/2),-90,1)
#imgRot = cv2.warpAffine(imgBW, matrix, (height, width))

### using Matplotlib
#plt.imshow(imgProc, cmap='gray')
#plt.xticks([])
#plt.yticks([])

plt.figure()
plt.subplot(2,2,1); plt.imshow(imgProc, cmap='gray')
plt.subplot(2,2,2); plt.imshow(Intf, cmap='gray')
plt.subplot(2,2,3); plt.plot(X,imgBW[int(h/2),:])
plt.subplot(2,2,4); plt.plot(Y,Intb)

plt.show()

plt.plot(Y,Intb)

plt.show()