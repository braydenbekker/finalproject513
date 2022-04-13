import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import mcGrid
from matplotlib.colors import ListedColormap




def visualize(randomGrid, colormap):
    keys = list(colormap.keys())
    colors = [colormap[key] for key in keys]

    image = [[keys.index(val) for val in row[1:]] for row in randomGrid]
    return image, colors




plt.close('all')




xmu = 0.5
xsigma = 0.1
xDistribution = lambda x: np.exp(-((x - xmu) / xsigma)**2 / 2) * 2.5 / np.sqrt(2 * np.pi)

ymu = 0.5
ysigma = 0.1
yDistribution = lambda y: np.exp(-((y - ymu) / ysigma)**2 / 2) * 2.5 / np.sqrt(2 * np.pi)


nrow = 100
ncol = 100




# X is a nrow*ncol by 2 matrix containing the positions of every grid point
X = np.array([])
for i in range(nrow):
    for j in range(ncol):
        X = np.append(X, [[i], [j]])
X = np.reshape(X, (int(ncol*nrow), 2))


newGrid = mcGrid.createGrid(nrow, ncol, xDistribution, yDistribution)

# labels is a flattened version of newGrid
labels = np.ravel(np.ndarray.tolist(newGrid.flatten()))





# Defaulat kernel is 'rbf'
clf = svm.SVC(C = 1.0)
clf.fit(X, labels)

labels = np.reshape(labels, (nrow, ncol))

predictions = []
for i in range(nrow):
    for j in range(ncol):
        predictions = np.append(predictions, clf.predict([[i,j]]))

predictions = np.reshape(predictions, (nrow,ncol))




colormap = {0:'white', 1:'red'}

image1, colors1 = visualize(predictions, colormap)
image2, colors2 = visualize(labels, colormap)

plt.figure(1)
plt.imshow(image1, cmap = ListedColormap(colors1), interpolation = 'nearest')
plt.title('Decision Boundary Prediction')
plt.grid()
plt.show()

plt.figure(2)
plt.imshow(image2, cmap = ListedColormap(colors2), interpolation = 'nearest')
plt.title('Random Monte Carlo Grid')
plt.grid()
plt.show()























