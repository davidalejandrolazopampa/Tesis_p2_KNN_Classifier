import numpy as np
from scipy.spatial import KDTree
x, y = np.mgrid[0:5, 2:8]
df = np.c_[x.ravel(), y.ravel()]
tree = KDTree(df)
# v id
# v 128
print(tree.data)
query = [0, 0]
dd, ii = tree.query([query], k=3)
print(dd, ii, sep='\n')
