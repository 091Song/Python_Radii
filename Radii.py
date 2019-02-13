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
# for interpolation
# https://docs.scipy.org/doc/scipy/reference/generated/
# scipy.optimize.curve_fit.html
#import scipy.optimize as opt
from scipy import optimize as sciopt
#from scipy.optimize import curve_fit

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

### scale
# nm/pixel
spix = 902.2

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
# ldep = 100

# Initial interpolation: interface positions
for i in range(2, h):
    # raw interpolation
    Int0[i] = pd.Series(imgBW[i,:]).idxmin()
           
    #######
    # using depth works fine typically
    # copy local BW depth (i.e. at i)
    # BWdepth = imgBW[i,:]
    # initial check: color depth for interface
    #######
    
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
        
# so far the Intb array saves interface positions
    
# Tune interface interpolation
# manually set sr for interface reevaluation
sr = 10
for i in range(sr+1, h-sr-1):
    low = np.mean(Intb[ int(i-1-sr) : int(i-1)])
    high = np.mean(Intb[ int(i+1) : int(i+1+sr)])
        
    # swap correctly
    if (low > high):
        low, high = high, low
    
    diff = int(high - low)
    
    # reevaluate interface positions
    # manually set the serching range as [low -5, high + 5]
    if (diff > 0):
        Intb[i] = \
        pd.Series(imgBW[i,int(low-5):int(high+5)]).idxmin() + int(low-5)

# Currently works fine
# for the reevaluation, possible to use a local 
# minimum depth near the interface (future development)
        
## before fitting a curve
## find tips        

## most advanced tip
tipa = Intb.min()

## searching limt
trange = tipa + 20

# to save local minimums
Lmin = np.zeros( (0,2) )

# index
# subtract data near boundaries
# 
idx = 50
# steps 
steps = 1


while (idx < h):
    #temporal values
    locx = 0.
    locy = 0.
    
    # specify the range
    if ( Intb[idx] < trange):
        # if (2.*Intb[idx] <= (Intb[idx-1] + Intb[idx+1]) ):
        if ( Intb[idx] < Intb[idx-1] and Intb[idx] < Intb[idx+1] ):
            
            Lmin = np.append(Lmin, [ [ idx, Intb[idx] ] ] , axis = 0 )
            
        elif ( Intb[idx] == Intb[idx+1] ):
            while ( Intb[idx] == Intb[idx+steps] ):
                steps += 1
            
            Lmin = np.append(Lmin, [ [ idx + 0.5*(steps-1), Intb[idx] ] ] ,\
                                      axis = 0 )
            
    
    idx = idx + steps
    steps = 1
    

# so far local mimimums were saved. 
# to save tip information
Tips = np.zeros( (0,2) )

for i in range(1,len(Lmin)-1):
    
    if ( Lmin[i,1] < Lmin[i-1,1] and Lmin[i,1] < Lmin[i+1,1] ):
        Tips = np.append(Tips, [ Lmin[i,:] ] , axis = 0 )


# behind this point Lmin array is not necessary
# del Lmin
### del(Lmin) --> use for radius evalution

# number of tips
tn = len(Tips)
# index number of the most advanced tip
idx_at = pd.Series(Tips[:,1]).idxmin()

#####
# from this point for interpolation
#####        

def LIMITS( ARR, tval, i0 = 0, steps = +1):
    # this function will find an idex of one value in ARR
    # the value should be closest to the tval.
    # the value shold also be equal to or lower than tval.
    # output index
    
    # upper lim
    larr = len(ARR)
        
    # target index
    tidx = i0
    
    # estimation
    if (steps == 0):
        print("a step value cannot be 0.")
        return 0
    elif ( steps > 0 and i0 > larr ):
        return 0
    elif (steps < 0 and i0 < 0 ):
        return 0
    else: 
        # search for the target idx
        while ( ARR[tidx] <= tval and tidx >= 0 and tidx <= larr ):
            tidx += steps
        
        return ( tidx - int( steps/np.abs(steps) ) )
    


# function def
def QuadEq(x, a, b, c):
    return a * (x**2) + b * x + c 
    


# diffusion length 
ld = 270./4.


# tips 

# save fitting parameters and ranges of a tip
Fparams = np.zeros( (tn,5) )

for i in range(0, tn):
    
    # find lower/upper limits for an interpolation
    
    # tips
    xtip = Tips[i,0]
    ytip = Tips[i,1]

    # an index for a lower limit
    idxl = LIMITS( Intb, (ytip+ld), int(xtip), -1) 
    # an index for a higher upper limit
    idxh = LIMITS( Intb, (ytip+ld), int(xtip) )
        
    # need to printout these values
    # print(i, idxl, Intb[idxl], idxu, Intb[idxu], ytip+ld)
    # popt, pcov = curve_fit(QuadEq, Y[idxl:idxu], Intb[idxl:idxu] )
    
    popt, pcov = sciopt.curve_fit(QuadEq, Y[idxl:idxh], Intb[idxl:idxh] )
    
    # save data
    # index of the lower limit
    Fparams[i,0] = idxl
    # index of the higer limit
    Fparams[i,1] = idxh
    # first parameter a
    Fparams[i,2] = popt[0]
    # second parameter b
    Fparams[i,3] = popt[1]
    # third parameter c
    Fparams[i,4] = popt[2]

# 

# rearrange interface positions
for i in range(0, h):
    # set white for a boundary 
    Intf[i,int(Intb[i])] = 255
    # interface : temporal 
    # Intb[i] = w - Intb[i]

        
    
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
#plt.plot(Y, Int0, 'k', Y,Intb, 'b', Y,Intbt, 'r')

#plt.plot(Y[600:800],Intb[600:800], 'b')
#plt.plot(Y, Int0, 'k', Y,Intb, 'b--')

plt.plot(Y,Intb, 'b')
plt.plot(Tips[:,0], Tips[:,1], 'rx')

idxl = int(Fparams[0,0])
idxh = int(Fparams[0,1])

# idx=0
for idx in range (0, tn):
    plt.plot( Y[int(Fparams[idx,0]):int(Fparams[idx,1])], \
                QuadEq(Y[int(Fparams[idx,0]):int(Fparams[idx,1])], \
                         Fparams[idx,2], Fparams[idx,3], Fparams[idx,4]), \
                         'r--')
                
#plt.plot( Lmin[idxl:idxh,0], QuadEq(Lmin[idxl:idxh,0], *popt), 'g-')
#plt.plot( Y[idxl:idxu], QuadEq( Y[idxl:idxu], *popt), 'g--')
#plt.ylim(290,350)

#chkl = 450
#chkr = 500
#plt.plot(Y[chkl:chkr], Int0[chkl:chkr], 'k', Y[chkl:chkr],Intb[chkl:chkr], 'r.')
#plt.plot(Y, Intb, 'b')
plt.ylim(top = 500)
plt.show()

#plt.plot( Y[idxl:idxu], QuadEq( Y[idxl:idxu], *popt), 'g--')
#plt.ylim(-250,250) #https://plot.ly/matplotlib/axes/
#plt.plot(Y[1:len(Y)]-0.5, (Intb[1:len(Intb)] - Intb[0:len(Intb)-1]))
#plt.plot(Y[1:chkr+1], (Intb[2:chkr+2] - Intb[0:chkr]))
#plt.show()

#plt.plot(X,imgBW[475,:]) #, X, BWdepth)
plt.plot( Lmin[:,0], Lmin[:,1], 'g-')

for idx in range (0, tn):
    plt.plot( Y[int(Fparams[idx,0]):int(Fparams[idx,1])], \
                QuadEq(Y[int(Fparams[idx,0]):int(Fparams[idx,1])], \
                         Fparams[idx,2], Fparams[idx,3], Fparams[idx,4]), \
                         'r--')

plt.ylim(top = 320)
plt.show()

for idx in range(0, tn):
    rad = 1./(2. * Fparams[idx,2])
    # previously calculated in unit of [pixel] spix = 902.2
    print("radius of tip {:d} = {:.3f}".format(idx, rad * spix/1000.) )
