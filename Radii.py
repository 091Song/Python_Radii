# -*- coding: utf-8 -*-
"""
Radius calculation using a sample image
 - Input: an interface image seen 
         from the side of the sample

 - Input parameters: 
     IFname: image file name
     spix: nm/pixel (written in an image)
     sr: searching range for interface positions
     ub: upper boundary
     msr: searching range for refining interface positions
     trange: searching range for cell tips
     ld: diffusion length [microns]

 - Output: Gray image (converted from the original)

@author: Y.Song
"""

############################################################
############################################################
############################################################
### import modules
# OpenCV 
import cv2
# Matplotlib
import matplotlib.pyplot as plt
# for calculation
import numpy as np 
# for data processing
import pandas as pd
# for interpolation
from scipy import optimize as sciopt
############################################################
############################################################
############################################################

############################################################
### Read an image
# image file name
IFname = 'SampleImage.jpg'

# Read an image as gray color
imgBW = cv2.imread(IFname, 0)

### Show images
# cv2.imshow('BlackWhite', imgBW)
# cv2.waitKey(0) temporal consideration
# cv2.waitKey(5)
# cv2.destroyAllWindows()

### save the gray image
#cv2.imwrite('SampleGray.jpg', imgBW)
cv2.imwrite('Gray'+IFname, imgBW)

### (h, w): height and width of an image
h, w = imgBW.shape 

############################################################

############################################################
### Input parameters
# scale: nm/pixel
spix = 902.2
# scale: microns/pixel
spix = spix/1000.

### search ranges 
sr = 100 # search limit
ub = 1100 # upper boundary
lb = sr

### diffusion length
ld = 270./4.

############################################################

############################################################
### functions
def LIMITS( ARR, tval, i0 = 0, steps = +1):
    # this function will find an idex of one value in an array of ARR
    # the value should be closest to the tval.
    # the value shold also be equal to or lower than tval.
    #
    # output: min/max index of ARR[tval] <= tval
    
    # upper lim
    larr = len(ARR)
        
    # target index
    tidx = i0
    
    # first estimation: out of bounds
    if (i0 < 0 or i0 > larr):
        print("Please check a tip position.")
        return 0
    
    # estimation
    if (steps == 0):
        print("a step value cannot be 0.")
        return 0
    elif (steps < 0) : 
        
        while (ARR[tidx] <= tval and tidx > 0 ) :
            tidx += steps
        
        tidx = tidx - steps if tidx > 0 else 0
        
        return tidx
    
    elif (steps > 0) :
        
        while (ARR[tidx] <= tval and tidx < larr ) :
            tidx += steps
        
        tidx = tidx - steps if tidx < larr else larr
        
        return tidx 
        
### quadratic eqution       
def QuadEq(x, a1, a2, a3):
    return a1 * (x**2) + a2 * x + a3

### 4th eq.
def EqOrder4(x, a1, a2, a3, a4, a5):
    return a1*(x**4) + a2*(x**3) + a3*(x**2) + a4*x + a5

############################################################
    
############################################################
### set an array for a horizontal axis 
X = np.arange(w)
Y = np.arange(h)

### (sample) depth map at the center of an image
fig1 = plt.figure()
plt.plot(X,imgBW[int(h/2),:])
plt.show()
fig1.savefig("Fig1.pdf")
###

### for interface locations in an image
Intb = np.zeros(h)

### intgerfaec interpolation
# check initial two values
# remind that (0,0) is the top left corner of an image
Intb[0] = pd.Series(imgBW[0,:]).idxmin()
Intb[1] = pd.Series(imgBW[0,:]).idxmin()

### initial interpolation: interface positions
### check a color depth: minimum for an interface
for i in range(2, h):    
    Intb[i] = pd.Series(imgBW[i,int(lb):int(ub)]).idxmin() + int(lb)
    
    # differences
    diffp = np.abs(Intb[i-2] - Intb[i-1])
    diffc = Intb[i-1] - Intb[i]
    # if using np.abs(diffc), it only catch one side of a tip parabola
    
    # refine searching limits
    # lim1 is close to the dendrite tip
    lim1 = Intb[i-1] - diffp - sr
    # lim2 is close to the groove (far away from a tip)
    lim2 = Intb[i-1] + diffp + sr
        
    # rearrange limits
    lim1 = lim1 if lim1 > sr else sr
    lim2 = lim2 if lim2 < ub else ub
        
    if ( lim2 == ub and diffc < 10 ):
        Intb[i] = ub
    elif (diffc > 200):
        # if the variation is too large (near the groove)
        # we set it as ub
        Intb[i] = ub
    else:
        Intb[i] = \
        pd.Series(imgBW[i,int(lim1):int(lim2)]).idxmin() + int(lim1)
        
# so far the Intb array saves interface positions

# Due to noise (black dots) of an image, 
# interpreted locations do not match the real interface sometimes.
# Need to refine the interface position (below)
    
### Refine interface positions
# reset sr for interface reevaluation
sr = int(sr/10)
# ignore boundaries
for i in range(sr+1, h-sr-1):
    # refine the searching limits 
    # using previous and next interface positions
    low = np.mean(Intb[ int(i-1-sr) : int(i-1)])
    high = np.mean(Intb[ int(i+1) : int(i+1+sr)])
        
    # swap correctly
    if (low > high):
        low, high = high, low
    
    diff = int(high - low)
    
    # reevaluate interface positions
    # manually set the serching range as [low -msr, high + msr]
    msr = 5
    if (diff > 0):
        Intb[i] = \
        pd.Series(imgBW[i,int(low-msr):int(high+msr)]).idxmin() + int(low-msr)

# Now, the Intb array has interface positions

### find cell tips

# for most advanced tip
tipa = Intb.min()

# searching limits
trange = tipa + 20

# For interface positiosn near tips
Lmin = np.zeros( (0,2) )

# index
idx = 50
# searching steps 
steps = 1

# Specify interface positions near tips
while (idx < h):
    #temporal values
    locx = 0.
    locy = 0.
    
    # specify the range
    if ( Intb[idx] < trange):
        
        if ( Intb[idx] == Intb[idx+1] ):
            # in case of same interface positions consecutively
            while ( Intb[idx] == Intb[idx+steps] ):
                steps += 1
            
            Lmin = np.append(Lmin, [ [ idx + 0.5*(steps-1), Intb[idx] ] ] ,\
                                      axis = 0 )
        else:
            # otherwise save the interfaces
            Lmin = np.append(Lmin, [ [ idx, Intb[idx] ] ] , axis = 0 )
        
    # update index
    idx = idx + steps
    # reset steps
    steps = 1
    
# so far, Lmin has refined interface positions near tips

# Save tip information
# for tip positions
Tips = np.zeros( (0,2) )

# find and save tips
for i in range(1,len(Lmin)-1):
    
    if ( Lmin[i,1] < Lmin[i-1,1] and Lmin[i,1] < Lmin[i+1,1] ):
        Tips = np.append(Tips, [ Lmin[i,:] ] , axis = 0 )

# number of tips
tn = len(Tips)
        
# Lmin can be deleted by
# del Lmin


##### Interpolation of a tip radius

# save fitting parameters and ranges of a tip
Fparams = np.zeros( (tn,5) )
# searching range
SearchR = np.zeros(tn)

# (1) use a parabola shape
for i in range(0, tn):
    
    # pre-determined tip positions
    xtip = Tips[i,0]
    ytip = Tips[i,1]
    
    # use a unit: pixel    
    # initially use a small range to calculation a tip radius
    sdist = 0.1 * ld/spix
    
    # an index for a lower limit
    idxl = LIMITS( Intb, (ytip+sdist), int(xtip), -1) 
    # an index for a upper limit
    idxh = LIMITS( Intb, (ytip+sdist), int(xtip) )
    
    # fitting a parabola curve
    popt, pcov = sciopt.curve_fit(QuadEq, Y[idxl:idxh+1], Intb[idxl:idxh+1] )
    # interpolation of a radius [pixel]
    rlocal = 1./(2. * popt[0] )
    
    # calculate a radius in different range
    # upto r ~ fitting distance from the tip
    
    # checking number
    ncheck = 0

    while ( sdist + 1. < rlocal and ncheck < 1000 ):
        # searching range
        sdist = sdist + 1. 
        
        idxl = LIMITS( Intb, (ytip + sdist), int(xtip), -1) 
        idxh = LIMITS( Intb, (ytip + sdist), int(xtip) )
        popt, pcov = sciopt.curve_fit(QuadEq, \
                                      Y[idxl:idxh+1], Intb[idxl:idxh+1] )
        rlocal = 1./(2. * popt[0] )
                
        ncheck += 1
    
    # save searching range
    # unit: microns
    SearchR[i] = sdist*spix
    
    # save data
    # index of the lower limit
    Fparams[i,0] = idxl
    # index of the higer limit
    Fparams[i,1] = idxh
    # first parameter a1
    Fparams[i,2] = popt[0]
    # second parameter a2
    Fparams[i,3] = popt[1]
    # third parameter a3
    Fparams[i,4] = popt[2]


# (2) use a tip radius (4th order equation)

# to save a tip radius
Rtips = np.zeros(tn)

for i in range(0, tn):
    
    # tips
    xtip = Tips[i,0]
    ytip = Tips[i,1]
    
    # an index for a lower limit
    idxl = LIMITS( Intb, (ytip+10), int(xtip), -1) 
    # an index for a higher upper limit
    idxh = LIMITS( Intb, (ytip+10), int(xtip) )
    
    # use 4th order for a curve fitting
    popt, pcov = sciopt.curve_fit(EqOrder4, Y[idxl:idxh+1], Intb[idxl:idxh+1])
    
    # derivatives
    dydx = 4.*popt[0]*pow(xtip,3) + 3.*popt[1]*pow(xtip,2) \
            + 2.*popt[2]*xtip + popt[3]        
    d2ydx2 = 12.*popt[0]*pow(xtip,2) + 6.*popt[1]*xtip + 2.*popt[2]
    
    # curvature
    curvk = d2ydx2 / pow( (1 + dydx*dydx),3./2. )
    
    # tip radius
    Rtips[i] = 1/curvk


############################################################
### Image outputs
    
# rearrange interface positions
#for i in range(0, h):
    # interface : temporal 
    # Intb[i] = w - Intb[i]

fig2 = plt.figure()

# interface
plt.plot(Y,Intb, 'k', linewidth = 1.5, label = 'Interface')

# tip positions
plt.plot(Tips[:,0], Tips[:,1], 'rx', linewidth = 1.5, markersize = 8,\
         label = 'Tips')

# parabola fit
for idx in range (0, tn):
    
    radd = 100
    idxl = int(Fparams[idx,0]) - radd
    idxr = int(Fparams[idx,1]+1) + radd
    
    plt.plot( Y[idxl:idxr], QuadEq(Y[idxl:idxr], \
             Fparams[idx,2], Fparams[idx,3], Fparams[idx,4]), \
            'r--', linewidth = 1.5)
    

plt.legend()

plt.ylim(top = 800)
plt.show()

fig2.savefig("Fig2.pdf")

### like the figure
fig3 = plt.figure()
# interface
plt.plot(Intb[:], h - Y[:], 'k', linewidth = 1.5, label = 'Interface')
# tip positinos
plt.plot(Tips[:,1], h-Tips[:,0], 'rx', linewidth = 1.5, markersize = 8,\
         label = 'Tips')

# parabola fit
for idx in range (0, tn):
    
    radd = 100
    idxl = int(Fparams[idx,0]) - radd
    idxr = int(Fparams[idx,1]+1) + radd
    
    plt.plot( QuadEq(Y[idxl:idxr], Fparams[idx,2], Fparams[idx,3], \
                     Fparams[idx,4]), \
            h - Y[idxl:idxr], 'r--', linewidth = 1.5)

plt.legend()
plt.show()


fig3.savefig("Fig3.pdf")

############################################################


############################################################
### save radius data
Fout = open('radii.dat', 'w')
# First line: total number of cells
Fout.write("#total number of cells: {:d}\n".format(tn))
# second line: labels
Fout.write("#tip for a tip radius (using polynomial function)\n")
Fout.write("#par for a parabola fit \n")
Fout.write("#units: microns \n")

#labels
for idx in range(0, tn):
    Fout.write("({:d})tip{:d}(tip)\t".format(idx+1, idx+1))

for idx in range(0, tn):
    Fout.write("({:d})tip{:d}(par)\t".format(idx+1+tn, idx+1))

Fout.write("\n")

# data
for idx in range(0, tn):
    rad = spix/(2. * Fparams[idx,2])
    Fout.write("{:.3f}\t".format(rad))

for idx in range(0, tn):
    rad = spix*Rtips[idx]
    Fout.write("{:.3f}\t".format(rad))

Fout.write("\n")

Fout.close()
############################################################
    
    

    
    
    
    