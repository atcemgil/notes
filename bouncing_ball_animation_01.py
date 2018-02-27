# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 01:30:34 2016

@author: cemgil

Bouncing Ball with x and y coordinates


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], frameon=True)
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])

global x
global y
global N 

# Number of particles to simulate
N = 1
x = np.zeros((2,N))
x[0,:] = np.random.rand(1,N)
x[1,:] = np.random.randn(1,N)/10
y = np.zeros((2,N))
y[0,:] = 0.9
y[1,:] = 0
ln, = plt.plot(x[0,:], y[0,:], 'go')
ln2, = plt.plot([x[0,0], 0], [y[0,0], 0], '--')

lnx, = plt.plot([x[0,0], x[0,0]], [y[0,0], 0], 'b--')
lny, = plt.plot([x[0,0], 0], [y[0,0], y[0,0]], 'r--')

global A
A = np.matrix('[1,0.05;0,1]')


def update(frame_number):
    global A
    global x
    global y
    global N
    #e = np.matrix('[0;1]')*0.002*np.random.randn(1,N)
    #e = 0
    x = A*x 
    y = A*y + np.matrix('[0;-1]')*0.005
    for i in range(N):
        if x[0,i]<0: 
            x[0,i]=0
            x[1,i]=-1.0*x[1,i]
        if x[0,i]>1:
            x[0,i] = 1
            x[1,i]=-1.0*x[1,i]
        if y[0,i]<0: 
            y[0,i]=0
            y[1,i]=-1.05*y[1,i]
        if y[0,i]>1:
            y[0,i] = 1
            y[1,i]=-0.2*y[1,i]

    ln.set_xdata(x[0,:].tolist()[0])
    ln2.set_xdata([x[0,0], 0])
    lnx.set_xdata([x[0,0], x[0,0]])
    lny.set_xdata([0, x[0,0]])
    #	ln.set_ydata(y[0,:])
    ln.set_ydata(y[0,:].tolist()[0])
    ln2.set_ydata([y[0,0], 0])
    lnx.set_ydata([0, y[0,0]])
    lny.set_ydata([y[0,0], y[0,0]])
    fig.gca().set_title(frame_number)
    return ln

animation = FuncAnimation(fig, update, interval=5)
plt.show()