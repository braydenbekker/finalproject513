

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

from sklearn.svm import SVC # "Support vector classifier"
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);






from math import *
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

num_points = 4000

sigma = .5;
mean = [0, 0]
cov = [[sigma**2,0],[0,sigma**2]]


x,y = np.random.multivariate_normal(mean,cov,num_points).T

svals = 16
fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0.2,0.2,0.8,0.8])
ax.scatter(x,y, s=svals, alpha=.1,cmap=cm.gray)


xi=[0,1,2,3,4,5]
np.random.choice(xi,len(xi),replace=False,)


import numpy as np
import numpy.random as npr

def getsurface(surf, si, sj, inds=None):
    """
    This function is called from a position (x,y) on the grid that is inside the
    decision boundary to test the remaining cardinal directions to extend the boundary.
    Parameters:
        si (int): The row index of the known grid point in the decision boundary.
        sj (int): The column index of the known grid point in the decision boundary.
        inds (list, tuple, int, int): Ongoing list of coordinates in the boundary.
    Functions:
        getsurface(si, sj, inds): recursively calls this function until the
            surface is complete.
    Returns:
        inds (list, tuple, int, int):
    """
    # Up, Down, Left, and Right indices from the input
    udlr=[(si+1,sj),(si-1,sj),(si,sj-1),(si,sj+1)]
    for ij in udlr:
        if surf[ij] == 1 and ij not in inds:
            inds.append(ij)
            inds = getsurface(surf, ij[0],ij[1], inds)
    return inds

nrow=7 # The number of rows in the input
ncol=9 # The number of columnss in the input
spins = np.ones((nrow,ncol), int)
print(spins)

spins = npr.randint(0, 2, (nrow,ncol)) * 2 - 1
print(spins)

#Center indices of the spins grid.
si,sj=int((nrow-1)/2),int((ncol-1)/2)
# Make Sure the center is a 1
spins[si,sj]=1

spins[0]=-1
spins[-1]=-1
spins[:,0]=-1
spins[:,-1]=-1



inds=getsurface(spins, si, sj, inds=[(si,sj)])

surf=np.zeros_like(spins)
for ind in inds:
    surf[ind] = 1

surf
