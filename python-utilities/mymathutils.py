#!/usr/bin/env python
# Filename: mymathutils.py

import numpy as np

#
# To do : Add options for how the edges of the data should be treated.  Zero-padding, reflecting, periodic, etc.
#
def smooth(x,window_len=11,window='hanning', kaiserpar=16.):
    """ Smooth a one-dimensional signal using the specified kernel function.
    
    Input :
    x - one-dimensional signal to smooth
    window_len - Overall size of window (in units of grid spacing)
    window - Name of window to use.  Allowed options are 'flat','hanning','hamming','bartlett','blackman','kaiser'
    kaiserpar - Parameter describing the Kaiser window.  Only has an effect if window='kaiser'

    Output :
    The smoothed signal obtained by convolving the input signal with the given filter
    """
    if x.ndim !=1:
        raise ValueError, "smooth only accepts 1-d arrays"
    if x.size < window_len:
        raise ValueError, "Cannot smooth data shorter than the window size"
    if window_len<3:
        return x
    if not window in ['flat','hanning','hamming','bartlett','blackman','kaiser']:
        raise ValueError, "Window is not of a recognized type"

# This is the correct choice for an odd sized window
# Is it correct for an even window?
    win_pad = window_len//2 + 1
    s=np.r_[x[win_pad-1:0:-1],x,x[-1:-win_pad:-1]]

    if window == 'flat':
        w=np.ones(window_len,'d')
    elif window == 'kaiser':
        w=eval('np.'+window+'(window_len, kaiserpar)')
    else:
        w=eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def _smooth(x,winLen=11,window='hanning', kaiserPar=16., boundary='reflect'):
    """Smooth a one-dimensional signal using a specified window function.  Options are given to choose a particular filter as well as how to treat boundary effects.

    Input:
    x - The signal to smooth
    winLen - Number of grid points in the window
    window - Name of the window
    kaiserPar - Kaiser window function parameter (only needed if window='kaiser')
    boundary - Determines how boundary effects are treated

    Output:
    The smoothed signal
    """
    if x.ndim != 1:
        raise ValueError, "_smooth only accepts 1d arrays"
    if x.size < winLen:
        raise ValueError, "Cannot smooth data shorter than the window size"
    if winLen < 3:
        return x

    allowedWindows = ['flat','hanning','hamming','bartlett','blackman','kaiser']
    if not window in allowedWindows:
        raise ValueError, "Window is not of a recognized type"

    allowedBoundary = ['reflect','constant','periodic','none']
    if not boundary in allowedBoundary:
        raise ValueError, "Choice of boundary effect handling not specified"

    winPad = winLen // 2
    
    return y

def _constructWindow(winName,winLen,kaiserPar=16.):
    """Return a window function of given length specified by a valid name

    Input:
    winName - Name of the window function.  Choose from 'flat','hanning','hamming','bartlett','blackman','kaiser'
    winLen - Length of the filter
    kaiserPar - Kaiser filter parameter.  Only used if winName='kaiser'.

    Output:
    The desired window function
    """

    if winName=='flat':
        kernel = np.ones(winLen,'d')
    else:  # Fix this from here down
        kernel = None
    return kernel / np.sum(kernel)

def savitzky_golay(y, window_size, order, deriv=0):
    """ Savitzky-Golay filter a signal

    Input :
    y : The signal to smooth
    

    Output :
    The signal with desired filter applied
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be integers")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size must be positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for polynomial order")

    order_range = range(order+1)
    half_window = (window_size - 1) // 2
# precompute coeffs
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv]
# pad the signal, using mirror- images
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve( m, y, mode='valid')

#
# Interpolation, etc
#
def extractConstantSurface(variable, varVal,increasing=True,tAxis=0):
    """Find the first occurance of varVal in variable at each point along an array
    
    Currently, will fail unless tAxis=0

    Input:
    variable - Array 2-D numpy array storing the value of a scalar variable at each point in space and time
    varVal - Variable value to extract hypersurface at
    increasing - Boolean.  If True then data values initially increase
    tAxis - 0 or 1.  Axis along which the time varies

    Output:
    A pair of numpy arrays stored as a list.  The first array stores the index at which
    """
    if variable.ndim != 2:
        raise ValueError, "variable must be a 2D numpy array storing the values of a scalar field"

    if tAxis != 0:
        raise ValueError, "Error in extractConstantSurface.  tAxis must be 0 in current implementation"

    if increasing:
        tInd = np.argmax(variable > varVal, axis=tAxis)
    else:
        tInd = np.argmax(variable < varVal, axis=tAxis)

    xInd = np.arange(len(variable[0]))
    dsFrac = (varVal - variable[tInd-1,xInd]) / (variable[tInd,xInd]-variable[tInd-1,xInd])

    return [ np.vstack((tInd,xInd)),dsFrac ]

# Add in support to also specify the axis along which x varies and let variable be a collection of scalars/vector/etc. instead of a single scalar
def evaluateOnSurface(variable,sliceParams,tAxis=0):
    """ Evaluate variable along the hypersurface specified by 
    
    Input:
    variable - Numpy array storing the variable
    sliceParams - Output of extractConstantSurface
    tAxis - Array axis along which time varies

    Output:
    The scalar variable evaluated on the constant surface specified by sliceParams
    """
    if variable.ndim != 2:
        raise ValueError, "Variable must be a 2D numpy array representing a scalar"
    if tAxis != 0:
        raise ValueError, "Error in evaluateOnSurface.  tAxis must be 0 in current implementation"

    dsFrac = sliceParams[1]
    tInd = sliceParams[0][0]
    xInd = sliceParams[0][1]
    varSlice = dsFrac*variable[tInd,xInd] + (1.-dsFrac)*variable[tInd-1,xInd]
    return varSlice

def evaluateConstantSlice(var,sliceVar,sliceVals,tAxis=0):
    """ Evaluate variable var on constant surfaces of sliceVar given by sliceVals

    Input:
    

    Output:
    """
    ind, frac = extractConstantSurface(sliceVar,sliceVals)
    return frac*var[ind[0],ind[1]] + (1.-frac)*var[ind[0]-1,ind[1]]


#
# Differential operators
#

def derivative(time, data):
    deriv=[]
    deriv.append((data[1]-data[0])/(time[1]-time[0]))
    for i in range(1,len(data)-1):
        deriv.append((data[i+1]-data[i-1])/(time[i+1]-time[i-1]))

    last = len(data)
    deriv.append((data[last-1]-data[last-2])/(time[last-1]-time[last-2]))

    return deriv

from numpy.fft import fftfreq, rfft, irfft
def derivativeSpectral(f,dk,order):
    nlat = len(f[0])
    kvals = fftfreq(nlat)*nlat*dk  # check why this norm is here
    fk = rfft(f,axis=1)
    fk = (1j*kvals[np.newaxis,:len(fk[0])])**order*fk
    dnf = irfft(fk,axis=1)
    return dnf

def interpolatePeriodicData(f,factor):
    nlat = len(f)*factor
    nk = nlat//2 + 1
    fk = rfft(f)
    nkPad = nk - len(fk)
    fk = np.concatenate( [fk,np.zeros(nkPad)] )
    fRefined = irfft(fk) # Add normalization in here

    return fRefined
    
