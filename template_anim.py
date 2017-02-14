%matplotlib inline
import time
import numpy as np
import pylab as plt
from IPython.display import display, clear_output

fig = plt.figure(figsize=(5,5))
ax = plt.gca()
ax.set_xlim((-2,2))
ax.set_ylim((-2,2))

th = np.pi/90.
A = 0.999*np.mat([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
X = np.mat([[1,-1],[1,-1]])

ln = plt.Line2D(xdata=(X[0,0], X[0,1]), ydata=(X[1,0], X[1,1]), marker='o',linewidth=1)
ax.add_line(ln)
ax.set_axis_off()
plt.close(fig)

for i in range(500):
    X = A*X
    ln.set_xdata((X[0,0], X[0,1]))
    ln.set_ydata((X[1,0], X[1,1]))
    display(fig)
    #display(plt.gcf())
    time.sleep(0.05)
    clear_output(wait=True)
