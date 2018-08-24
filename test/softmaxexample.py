import numpy as np

scores = [3.0,1.0,0.2]

def softmax(x):
    """Compute softmax value for X. """
    return np.exp(x)/ np.sum(np.exp(x), axis=0)

print(softmax(scores))


#plot the softmax curves#

import matplotlib.pyplot as pl

x = np.arange(-2.0, 6.0, 0.1)
# print(x)
# scores = np.vstack([x,np.ones_like(x),0.2*np.ones_like(x)])
scores = np.array([x, np.ones_like(x), 0.2*np.ones_like(x)])
# print(scores)
pl.plot(x, softmax(scores).T, linewidth=2)
pl.show()