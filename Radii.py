# -*- coding: utf-8 -*-
"""
    
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

Int0 = np.zeros(h)
Intb = np.zeros(h)
Intbt = np.zeros(h) ## temporal - comparison

# for local depth 
BWdepth = np.zeros(w)

# check initial two values
Int0[0] = pd.Series(imgBW[0,:]).idxmin()
Intb[0] = pd.Series(imgBW[0,:]).idxmin()

Int0[1] = pd.Series(imgBW[0,:]).idxmin()
Intb[1] = pd.Series(imgBW[0,:]).idxmin()

# temporal
Intbt[0] = Intb[0] 
Intbt[1] = Intb[1] 

# search ranges 
sr = 100 # search limit
ub = 1100 # upper boundary
lb = sr

# use color depth
ldep = 100

# check the darkest point at a height i 
# use differences
for i in range(2, h):
    # raw interpolation
    Int0[i] = pd.Series(imgBW[i,:]).idxmin()
    
    ###
    '''
    cdep = imgBW[i,Int0[i]]
    # differences
    diff = np.abs(Int0[i] - Int0[i-1])
    
    # interface positions 
    # Intb[i] = pd.Series(imgBW[i,lw:ub]).idxmin() + lw
    
    
    #if (diff < 0.5 * w ):
    # sort step 1: using difference to 
    if ( diff < 0.5 * w and cdep < ldep):
        # ignore data below ub
        Intb[i] = pd.Series(imgBW[i,lb:ub]).idxmin() + lb
        
    else:
        Intb[i] = ub #Intb[i-1]
       '''
       
    #######
    # using depth works fine typically
    # copy local BW depth (i.e. at i)
    # BWdepth = imgBW[i,:]
    # initial check: color depth for interface
    Intb[i] = pd.Series(imgBW[i,int(lb):int(ub)]).idxmin() + int(lb)
    
    # differences
    diffp = Intb[i-2] - Intb[i-1]
    diffc = Intb[i-1] - Intb[i]
    
    if (diffp < 0) :
        # lim1 is close to the dendrite tip
        # lim2 is close to the groove (far away from a tip)
        lim1 = Intb[i-1] + diffp - sr
        lim2 = Intb[i-1] - diffp + sr
    else:
        lim1 = Intb[i-1] - diffp - sr
        lim2 = Intb[i-1] + diffp + sr
    
    # rearrange limits
    lim1 = lim1 if lim1 > sr else sr
    lim2 = lim2 if lim2 < ub else ub
    
    #if (Intbt[i-1] < )
    
    if ( lim2 == ub and diffc < 10 ):
        Intb[i] = ub
    elif (diffc > 200):
        # if the variation is too large
        Intb[i] = ub
    else:
        Intb[i] = \
        pd.Series(imgBW[i,int(lim1):int(lim2)]).idxmin() + int(lim1)
        
        
    
        
# rearrange interface positions
for i in range(0, h):
    # set white for a boundary 
    Intf[i,int(Intb[i])] = 255
    # interface : temporal 
    # Intb[i] = w - Intb[i]



########
# update later
    # if there are two data points which have same minimums
    # ignore the noisy points
    
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
plt.plot(Y, Int0, 'k', Y,Intb, 'b', Y,Intbt, 'r')

#plt.plot(Y, Int0, 'k', Y,Intb, 'b--')

chkr = 600
#plt.plot(Y[0:chkr], Int0[0:chkr], 'k', Y[0:chkr],Intbt[0:chkr], 'r--')
#plt.plot(Y, Intb, 'b')
plt.show()

plt.ylim(-250,250) #https://plot.ly/matplotlib/axes/
#plt.plot(Y[1:len(Y)]-0.5, (Intb[1:len(Intb)] - Intb[0:len(Intb)-1]))
plt.plot(Y[1:chkr+1], (Intb[2:chkr+2] - Intb[0:chkr]))
plt.show()

plt.plot(X,imgBW[100,:]) #, X, BWdepth)
plt.show()