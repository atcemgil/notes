
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt


def norm_ball(p):
    step = np.pi/128
    THETA = np.arange(0, 2*np.pi+step, step)
    X = np.mat(np.zeros((2,len(THETA))))
    for i, theta in enumerate(THETA):
        x = (np.cos(theta), np.sin(theta))
        a = (1/(np.abs(x[0])**p + np.abs(x[1])**p ))**(1/p)
        X[:, i] = a*np.mat(x).T
        
    return X


P = np.arange(0.25,5.25,0.25)
#print(X)
fig = plt.figure(figsize=(10,10))
NumPlotRows = 5
NumPlotCols = 4

for i,p in enumerate(P):

    X = norm_ball(p=p)
    plt.subplot(NumPlotRows, NumPlotCols, i+1)
    plt.plot(X[0,:].T, X[1,:].T,'-',clip_on=False)
    ax = fig.gca()
    ax.set_xlim((-2,2))
    ax.set_ylim((-2,2))
    ax.axis('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #plt.plot(X[0,:].tolist())
    for loc, spine in ax.spines.items():
        spine.set_color('none')  # don't draw spine

    plt.title(p)
    
plt.show()