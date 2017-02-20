# Demonstrates the image of a normball 
# under a linear transformation matrix 
#

import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML
from matplotlib import rc

from notes_utilities import pnorm_ball_points
from notes_utilities import bmatrix



rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

fig = plt.figure(figsize=(10,5))

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

dx,dy = pnorm_ball_points()
ln_domain_ball = plt.Line2D(dx,dy,markeredgecolor='k', linewidth=1, color='b')
ln_e1 = plt.Line2D([0,1],[0,0],markeredgecolor='k', marker='o', color='b')
ln_e2 = plt.Line2D([0,0],[0,1],markeredgecolor='k', markerfacecolor='w', marker='o', color='k')

dx,dy = pnorm_ball_points()
ln_measure_ball = plt.Line2D(dx,dy,markeredgecolor='k', linewidth=1, color='k',linestyle=':')

dx,dy = pnorm_ball_points()
ln_range_ball = plt.Line2D(dx,dy,markeredgecolor='k', linewidth=1, color='b',linestyle='-')
ln_e1range = plt.Line2D([0,1],[0,0],markeredgecolor='k', marker='o', color='b')
ln_e2range = plt.Line2D([0,0],[0,1],markeredgecolor='k', markerfacecolor='w', marker='o', color='k')

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_xlim((-4,4))
ax1.set_ylim((-4,4))

ax2.set_xlabel('$y_1$')
ax2.set_ylabel('$y_2$')
ax2.set_xlim((-4,4))
ax2.set_ylim((-4,4))

ax1.add_line(ln_domain_ball)
ax1.add_line(ln_e1)
ax1.add_line(ln_e2)

ax2.add_line(ln_range_ball)
ax2.add_line(ln_measure_ball)
txt = ax2.text(-1,-3,'$\left(\right)$',fontsize=15)
txtr = ax2.text(1,1,'$\left(\right)$',fontsize=15)

ax2.add_line(ln_e1range)
ax2.add_line(ln_e2range)

plt.close(fig)

MAX_p = 5
MAX_q = 5

def set_line(a_11=1, a_21=0, a_12=0, a_22=1, q=2, r=2, p=2):
    A = np.array([[a_11, a_12],[a_21, a_22]])
    if p == MAX_p:
    	p = np.inf
    if q == MAX_q:
    	q = np.inf

    #S = A.dot(A.T)
    dx,dy = pnorm_ball_points(np.eye(2),p=q)
    ln_domain_ball.set_xdata(dx)
    ln_domain_ball.set_ydata(dy)
    dx,dy = pnorm_ball_points(A,p=q)
    ln_range_ball.set_xdata(dx)
    ln_range_ball.set_ydata(dy)
    ln_e1range.set_xdata(np.c_[0,A[0,0]])
    ln_e1range.set_ydata(np.c_[0,A[1,0]])
    ln_e2range.set_xdata(np.c_[0,A[0,1]])
    ln_e2range.set_ydata(np.c_[0,A[1,1]])

    dx,dy = pnorm_ball_points(r*np.eye(2),p=p)
    ln_measure_ball.set_xdata(dx)
    ln_measure_ball.set_ydata(dy)
    
    txt.set_text(bmatrix(A))
    txtr.set_text(r)
    txtr.set_x(r)
    txtr.set_y(r)
    
    display(fig)
    #ax.set_axis_off()
    
interact(set_line, a_11=(-2,2,0.01), a_12=(-2, 2, 0.01), a_21=(-2, 2, 0.01), a_22=(-2, 2, 0.01), q=(0.1,MAX_q,0.1), r=(0.2,5,0.1), p=(0.1,MAX_p,0.1))


