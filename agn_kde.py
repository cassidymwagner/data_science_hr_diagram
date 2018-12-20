#!/usr/bin/env python
# coding: utf-8

# # Homework 7
# 
# ## ASTR 5900, Fall 2017, University of Oklahoma
# 
# ### KMeans and KDE

# # Problem 1
# 
# ### Part A
# 
# Load the Old Faithful data from http://www.stat.cmu.edu/~larry/all-of-statistics/=data/faithful.dat.  
# 
# A standard procedure in problems with multi-dimensional data is to standardize the data, or give each dimension the same scaling.  It is common to make every parameter distributed around 0 with a standard variance.  That is, find a new data set with parameters $y_{i}^{(j)}$ where:
# 
# $$ y_{i}^{(j)} = \frac{x_{i}^{(j)} - \mu^{(j)}}{\sigma^{(j)}}$$
# 
# Here $x_{i}^{(j)}$ is the $i$th data point in the $j$th dimension.
# 
# Transform the Old Faithful Data in this manner.

# In[1]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import colors
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 10)


# In[2]:


data = np.loadtxt("faithful.dat", skiprows=26)
eruptions=data[:,1]
waiting=data[:,2]


# In[3]:


eruptions=(eruptions-eruptions.mean())/eruptions.std()
waiting=(waiting-waiting.mean())/waiting.std()


# ### Part B
# 
# Use `scipy.cluster.vq.kmeans` or `sklearn.cluster.KMeans` to split the Old Faithful data into 2 clusters.  Plot the data with each point colored according to its cluster label.  Also plot the centers of the clusters by making it distinguishable from the data.
# 
# Read more at http://scikit-learn.org/stable/modules/density.html.  Consider looking at the examples on the `scikit-learn` website and the lecture.

# In[4]:


from scipy.cluster.vq import kmeans,vq,whiten


# In[5]:


temp=np.vstack((eruptions,waiting))
properties=temp.T

centroids,_ = kmeans(properties,2)
idx,_ = vq(properties,centroids)


# In[6]:


plt.plot(eruptions[idx==0],waiting[idx==0],'go')
plt.plot(eruptions[idx==0].mean(),waiting[idx==0].mean(),'ro',markersize=10)
plt.plot(eruptions[idx==1],waiting[idx==1],'bo')
plt.plot(eruptions[idx==1].mean(),waiting[idx==1].mean(),'ro',markersize=10)


# ### Part C
# 
# Code from scratch (that is, with default python and `numpy` only) your own k-means clustering algorithm to split the data into 2 clusters.  Refer to the lecture and Figure 9.1 in Bishop.  Plot each step in the process

# In[7]:


data=np.column_stack((eruptions,waiting))
xdata=eruptions
ydata=waiting

mu1=(np.random.uniform(eruptions.min(),eruptions.min()),np.random.uniform(waiting.min(),waiting.max()))
mu2=(np.random.uniform(eruptions.min(),eruptions.min()),np.random.uniform(waiting.min(),waiting.max()))

cluster1_data=np.zeros(len(data))
cluster2_data=np.zeros(len(data))
r_k=np.zeros(len(data))


# In[8]:


plt.scatter(xdata,ydata,c='b',marker='*')
plt.scatter(mu1[0],mu1[1],c='g',s=200)
plt.scatter(mu2[0],mu2[1],c='g',s=200)
plt.title("Before Kmeans Clustering")
plt.show()

for s in range(3):
    
    cluster1_xpop=[]
    cluster1_ypop=[]
    cluster2_xpop=[]
    cluster2_ypop=[]
    
    for i in range(len(data)):
        
        cluster1_data[i]=(xdata[i]-mu1[0])**2+(ydata[i]-mu1[1])**2

        cluster2_data[i]=(xdata[i]-mu2[0])**2+(ydata[i]-mu2[1])**2
        
        if cluster1_data[i]<cluster2_data[i]:
            r_k[i] = 0
        else:
            r_k[i]=1    
    
    for j in range(len(data)):
        
        if r_k[j] == 0:
            cluster1_xpop=np.append(cluster1_xpop,xdata[j])
            cluster1_ypop=np.append(cluster1_ypop,ydata[j])
        else:
            cluster2_xpop=np.append(cluster2_xpop,xdata[j])
            cluster2_ypop=np.append(cluster2_ypop,ydata[j])
    
    mu1 = (np.mean(cluster1_xpop),np.mean(cluster1_ypop))
    mu2 = (np.mean(cluster2_xpop),np.mean(cluster2_ypop))
    
    plt.subplot(2,3,s+1)
    plt.title("Iteration Number "+str(s+1))
    plt.scatter(cluster1_xpop,cluster1_ypop,c='b',marker='*')
    plt.scatter(cluster2_xpop,cluster2_ypop,c='r',marker='*')
    plt.scatter(mu1[0],mu1[1],c='g',s=200)
    plt.scatter(mu2[0],mu2[1],c='g',s=200)


# # Problem 2
# 
# In this problem you will perform kernel density estimation to produce
# an optimal representation of the Chandra X-ray observatory data from
# NGC 4636.  The X-ray emission traces the emission of hot gas in the galaxy.
# The data consist of a list of the positions of individual
# photons on the detector in sky coordinates.
# 
# ### Part A
# 
# Load the data of NGC 4636 from `ngc4636.dat`.
# 
# Create a plot showing the individual photon points using `matplotlib.pyplot`.
# 
# Plot a histogram of the data.  Experiment with the binsize and plot representation until you obtain a pleasing image of the galaxy.  Do you see any structure in the image
# besides the central concentration of hot gas?  Explain.
# 
# Perform KDE data using `sklearn.neighbors.kde.KernelDensity`.  Experiment with the 
# band width and kernel until you obtain a pleasing image of the galaxy.  Do you see any 
# structure in the image besides the central concentration of hot gas?  Explain.
# Note that KernelDensity.score_samples returns the log of the distribution.
# Also note that it may help for plotting the image to sample the distribution on a grid;
# refer to the example shown in class.

# In[9]:


xray_data=np.loadtxt('ngc4636.dat')
xcoord=xray_data[:,0]
ycoord=xray_data[:,1]


# In[10]:


counts,xedges,yedges,image=plt.hist2d(xcoord,ycoord,bins=100)


# #### Discussion: 
# It looks like the galaxy has weak spiral arms.

# In[11]:


import sklearn.neighbors.kde as kde


# In[12]:


def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = kde.KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde_skl.fit(xy_train)

    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


# In[13]:


xx,yy,zz=kde2D(xcoord,ycoord,10)


# In[14]:


plt.pcolormesh(xx,yy,zz)


# In[15]:


plt.pcolormesh(xx,yy,zz)
plt.scatter(xcoord,ycoord,s=2,color='white',alpha=.2)


# #### Discussion: 
# The spiral arms are more prominent and the elliptical structure of the galaxy is a bit more resolved.

# ### Part B
# 
# What is the optimal bandwidth to replicate the structure of the data?  One way to determine this value is to maximize the 'leave-one-out likelihood cross-validation' function:
# 
# $$ \text{CV}_l \, (h) = \frac{1}{N} \sum_{i=1}^{N} \log \hat{f}_{h, -i}(x_i)$$
# 
# where $\hat{f}_{h, -i}(x_i)$ is the estimated density at position $x_i$ with the $i$th data point left out and bandwith $h$.  Refer to the lecture.
# 
# You will want to create a 1D grid of 20 different values of h to test.  Examining the results
# of part A, what do you think the minimum binsize should be?  Explain.  What do you think
# the maximum binsize should be?  Explain.
# 
# In short, approximate the optimal bin size by finding the bin size (among a "good" sample of widths) that maximizes $CV_l$.
# 
# **NOTE: This calculation could take several minutes.  To test your code, you may wish to only consider a fraction of the galaxy data with only a few kernel widths.**
# 
# At the end of the day:
# 1. Print your optimal $h$  
# 2. Plot $CV_l$ versus the your $h$ samples (perhaps $\log h$)
# 3. Plot the estimated density of the galaxy using the optimal $h$

# #### Discussion: 
# By eye from part A, it seems as though the best binsize would be near 10, therefore the minimum binsize should be around 5 and the maximum around 15.

# In[ ]:


temp=np.linspace(0.68,1.2,20)
h_candidates=np.array(10.0**temp)
cv_other=[]
cv=np.zeros([h_candidates.shape[0],xray_data.shape[0]])
numh=h_candidates.shape[0]


# In[ ]:


for i in range(numh):
    htemp=h_candidates[i]
    kde_skl = kde.KernelDensity(bandwidth = htemp, kernel='gaussian')
    numpnts=xray_data.shape[0]
    for j in range(20,numpnts-20):
        data_one_out=np.concatenate((xray_data[0:j],xray_data[j+1:numpnts]))
        kde_skl.fit(data_one_out)
        log_pdf = kde_skl.score_samples(xray_data[j].reshape(1,-1))
        cv[i,j]=cv[i,j]+log_pdf[0]


# In[ ]:


### Uncomment if dumping into pickle file ####
# import pickle
# object=cv
# file=open('hw7_part2b_data','wb')
# pickle.dump(object,file)
# file.close()


# In[ ]:


#### Uncomment if loading pickle file ####
# file=open('hw7_part2b_data','rb')
# cv=pickle.load(file)


# In[ ]:


cv_out=np.zeros(20)
for i in range(20):
    cv_out[i]=cv[i,:].sum()


# In[ ]:


plt.semilogx(h_candidates,cv_out)
plt.xlabel('log h')
plt.ylabel('CV')


# In[ ]:


print('The value of h when CV is maximized',h_candidates[np.where(cv_out==cv_out.max())])


# In[ ]:


h=h_candidates[np.where(cv_out==cv_out.max())][0]


# In[ ]:


xx,yy,zz=kde2D(xcoord,ycoord,h)


# In[ ]:


plt.pcolormesh(xx,yy,zz)


# #### Discussion:
# The predicted best binsize was 10, and the optimal binsize is within the range of 5-15, relatively close to 10, resulting in a similar density estimation of the galaxy as in part A.

# In[ ]:




