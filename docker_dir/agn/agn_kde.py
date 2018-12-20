import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn.neighbors.kde as kde

xray_data=np.loadtxt('ngc4636.dat')
xcoord=xray_data[:,0]
ycoord=xray_data[:,1]

counts,xedges,yedges,image=plt.hist2d(xcoord,ycoord,bins=100)
plt.show()

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = kde.KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde_skl.fit(xy_train)

    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

xx,yy,zz=kde2D(xcoord,ycoord,10)

plt.pcolormesh(xx,yy,zz)
plt.show()


plt.pcolormesh(xx,yy,zz)
plt.scatter(xcoord,ycoord,s=2,color='white',alpha=.2)
plt.show()


