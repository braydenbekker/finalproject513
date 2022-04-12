import numpy as np

def createGrid(n, m, xDist, yDist):
    grid = np.array([[None] * m] * n)
    
    for i in range(n):
        for j in range(m):
            prob = xDist(i / n) * yDist(j / m)
            grid[i, j] = np.random.choice([0, 1], 1, replace = True, p = [1 - prob, prob])
    
    return grid