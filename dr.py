import matplotlib.pyplot as plt
import mcGrid
import numpy as np
from surface import getsurface

nrow = 100
ncol = 100

xmu = 0.5
xsigma = 0.1
xDistribution = lambda x: np.exp(-((x - xmu) / xsigma)**2 / 2) * 2.5 / np.sqrt(2 * np.pi)

ymu = 0.5
ysigma = 0.1
yDistribution = lambda y: np.exp(-((y - ymu) / ysigma)**2 / 2) * 2.5 / np.sqrt(2 * np.pi)

def getGrids():
    newGrid = mcGrid.createGrid(nrow, ncol, xDistribution, yDistribution)

    #Center indices of the spins grid.
    si,sj=int((nrow-1)/2),int((ncol-1)/2)

    # Make Sure the center is a 1
    newGrid[si,sj]=1

    newGrid[0]=0
    newGrid[-1]=0
    newGrid[:,0]=0
    newGrid[:,-1]=0

    inds=getsurface(newGrid, si, sj, inds=[(si,sj)])


    outGrid=np.zeros_like(newGrid)
    for i in inds:
        outGrid[i]=1

    return outGrid, newGrid

#The number of total grids to generate
numGrids=1000
#percentage for train, test, and validation
sztrain, sztest, szvalid = 0.6,0.3,0.2

trainGridin=np.zeros((int(numGrids*sztrain),nrow,ncol))
trainGridout=np.zeros((int(numGrids*sztrain),nrow,ncol))
for i in range(len(trainGridin)):
    outGrid, newGrid = getGrids()
    trainGridin[i]=newGrid
    trainGridout[i]=outGrid

with open("trainin.npy", 'wb') as f:
    np.save(f, trainGridin)
with open('trainout.npy', 'wb') as f:
    np.save(f, trainGridout)
del trainGridout, trainGridin

testGridin=np.zeros((int(numGrids*sztest), nrow, ncol))
testGridout=np.zeros((int(numGrids*sztest),nrow,ncol))
for i in range(len(testGridin)):
    outGrid, newGrid = getGrids()
    testGridin[i]=newGrid
    testGridout[i]=outGrid

with open("testin.npy", 'wb') as f:
    np.save(f, testGridin)
with open('testout.npy', 'wb') as f:
    np.save(f, testGridout)
del testGridout, testGridin

validGridin=np.zeros((int(numGrids*szvalid),nrow,ncol))
validGridout=np.zeros((int(numGrids*szvalid),nrow,ncol))
for i in range(len(validGridin)):
    outGrid, newGrid = getGrids()
    validGridin[i]=newGrid
    validGridout[i]=outGrid

with open("validin.npy", 'wb') as f:
    np.save(f, validGridin)
with open('validout.npy', 'wb') as f:
    np.save(f, validGridout)
del validGridout, validGridin
