import numpy as np
import mcGrid
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.close('all')

def visualize(randomGrid, colormap):
    keys = list(colormap.keys())
    colors = [colormap[key] for key in keys]

    image = [[keys.index(val) for val in row[1:]] for row in randomGrid]
    return image, colors




numRows = 100
numCols = 100

xmu = 0.5
xsigma = 0.1
xDistribution = lambda x: np.exp(-((x - xmu) / xsigma)**2 / 2) * 2.5 / np.sqrt(2 * np.pi)

ymu = 0.5
ysigma = 0.1
yDistribution = lambda y: np.exp(-((y - ymu) / ysigma)**2 / 2) * 2.5 / np.sqrt(2 * np.pi)

colormap = {0:'white', 1:'red'}




newGrid = mcGrid.createGrid(numRows, numCols, xDistribution, yDistribution)

image, colors = visualize(newGrid, colormap)

plt.imshow(image, cmap = ListedColormap(colors), interpolation = 'nearest')
plt.show()


