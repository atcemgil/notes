# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:08:26 2016

@author: cemgil
"""

import scipy as sc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation

from itertools import product

## Load data
df_arac = pd.read_csv(u'data/arac.csv',sep=';')

BaseYear = 1995
x = np.matrix(df_arac.Year[31:]).T-BaseYear
y = np.matrix(df_arac.Car[31:]).T/1000000

# Setup the vandermonde matrix
N = len(x)
A = np.hstack((np.ones((N,1)), x))

global w
# Starting point
w = np.matrix('[15; -6]')


## Compute Error Surface
left = -5
right = 25
bottom = -6
top = 6
step = 0.1
W0 = np.arange(left,right, step)
W1 = np.arange(bottom,top, step)

ErrSurf = np.zeros((len(W1),len(W0)))

for i,j in product(range(len(W1)), range(len(W0))):
	e = y - A*np.matrix([W0[j], W1[i]]).T
	ErrSurf[i,j] = e.T*e/2


# Create new Figure

fig = plt.figure(figsize=(10,10))
plt.imshow(ErrSurf, interpolation='nearest',
			vmin=0, vmax=1000,origin='lower',extent=(left,right,bottom,top))
plt.xlabel('w0')
plt.ylabel('w1')


# Learning rate: The following is the largest possible fixed rate for this problem
#eta = 0.000692
eta = 0.000672
ln, = plt.plot(w[0].tolist()[0], w[1].tolist()[0], 'ow')

def update(frame_number):
	global w
	e = y - A*w
	
	g = -A.T*e
	w = w - eta*g
	
	ln.set_xdata(w[0].tolist()[0])
	ln.set_ydata(w[1].tolist()[0])
	return ln

animation = FuncAnimation(fig, update, interval=10)
plt.show()