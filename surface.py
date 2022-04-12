import numpy as np

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
