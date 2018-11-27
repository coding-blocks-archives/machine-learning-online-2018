#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_circles
from matplotlib import pyplot as plt


# In[2]:


plt.style.available


# In[3]:




X,Y   = make_circles(n_samples=500,noise=0.02)
print(X.shape)


# In[4]:


styles = plt.style.available
print(styles)


# In[7]:


plt.style.use(styles[4])
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[10]:


X1 = X[:,0]
X2 = X[:,1]
X3 = X1**2 + X2**2
print(X3.shape)


# In[23]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, X3, zdir='z', s=20, c=Y, depthshade=True)


# In[26]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X1 Axis")
ax.set_ylabel("X2 Axis")
ax.set_zlabel("X3 Axis")

plt.show()




