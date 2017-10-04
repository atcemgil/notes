%matplotlib inline

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import clear_output, display, HTML
from matplotlib import rc

mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)

fig = plt.figure(figsize=(5,5))
ax = plt.gca()
ln = plt.Line2D([0],[0])

ax.add_line(ln)
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_axis_off()

plt.close(fig)

def set_line(th):
    ln.set_xdata([np.cos(th), -np.cos(th)])
    ln.set_ydata([np.sin(th), -np.sin(th)])
    display(fig)   
    
interact(set_line, th=(0.0, 2*np.pi,0.01))

