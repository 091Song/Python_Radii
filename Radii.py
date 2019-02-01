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

# Read an image
imgBW = cv2.imread('SampleImage.jpg', 0) # Black and White image

### Show images
#cv2.imshow('Color', imgCL)
cv2.imshow('BlackWhite', imgBW)
# cv2.waitKey(0) temporal consideration
cv2.waitKey(5)
cv2.destroyAllWindows()

### save an image
#cv2.imwrite('SampleGray.jpg', imgBW)

### (h, w): height and widht of an image
h, w = imgBW.shape 

# set an array for a horizontal axis 
X = np.arange(w)
Y = np.arange(h)

# for new array: same data type as in the image
Intf = np.zeros((h,w), dtype = imgBW.dtype )
Intb = np.zeros(h)
## temporal - comparison
Intbub = np.zeros(h)

# check initial two values
Intb[0] = pd.Series(imgBW[0,:]).idxmin()
# temporal
Intbub[0] = Intb[0] 

# search ranges 
sr = 100 # search limit
ub = w-sr # upper boundary

# check the darkest point at a height i 
# use differences
for i in range(1, h):
    # interface positions 
    Intb[i] = pd.Series(imgBW[i,:]).idxmin()
    # differences
    diff = np.abs(Intb[i] - Intb[i-1])
    
    # sort step 1: using difference to 
    if (diff < 0.5 * w):
        Intb[i] = pd.Series(imgBW[i,:]).idxmin()
    else:
        Intb[i] = Intb[i-1]
    
    # upper limit
    Intbub[i] = pd.Series(imgBW[i,:]).idxmin()
    ulim = Intbub[i-1] + sr if Intbub[i-1] < ub else ub
    
    if (diff < 0.5 * w):
        Intbub[i] = pd.Series(imgBW[i,0:ulim]).idxmin()
    else:
        Intbub[i] = Intbub[i-1]

    
    
    
    # due to the image processing
    
    
    # llim is close to the dendrite tip
    # ulim is close to the groove (far away from a tip)
    #llim = 100
    #Intb[i-1] - sr if Intb[i-1] > sr else 0 
    #ulim = Intb[i-1] + sr
    
            
    
# ignore values < 0 
# sort difference < 100 -> turn them 0
    
# rearrange interface positions
for i in range(0, h):
    # set white for a boundary 
    Intf[i,Intb[i]] = 255
    # interface : temporal 
    # Intb[i] = w - Intb[i]



########
# update later
    # if there are two data points which have same minimums
    # ignore the noisy points
    
    # use a function to obtain an initial value
########
        
    
### This is for check image 
# to save the original image, copy it
# imgProc = np.array(imgBW)
# imgProc[h/2-10:h/2+10, :] = np.uint8(255)

### sort
#imgBW2 = (imgBW < 120 )

#matrix = cv2.getRotationMatrix2D((height/2,width/2),-90,1)
#imgRot = cv2.warpAffine(imgBW, matrix, (height, width))

### using Matplotlib
#plt.imshow(imgProc, cmap='gray')
#plt.xticks([])
#plt.yticks([])

#plt.figure()
#plt.subplot(2,2,1); plt.imshow(imgProc, cmap='gray')
#plt.subplot(2,2,2); plt.imshow(Intf, cmap='gray')
#plt.subplot(2,2,3); plt.plot(X,imgBW[int(h/2),:])
#plt.subplot(2,2,4); plt.plot(Y,Intb)

#plt.show()
plt.plot(Y,Intb,Y,Intbub)

#plt.ylim(-250,250) #https://plot.ly/matplotlib/axes/
#plt.plot(Y[1:len(Y)]-0.5, (Intb[1:len(Intb)] - Intb[0:len(Intb)-1]))
plt.show()