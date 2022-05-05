#!/usr/bin/env python
# coding: utf-8

# In[2]:


# File Input / Output package
import numpy as np
from scipy import io as scipyIo
array = np.ones((4, 4))
scipyIo.savemat("example.mat", {'ar': array}) 
data = scipyIo.loadmat("example.mat", struct_as_record=True)
print(data)


# In[11]:


# cluster example
import numpy as np
from scipy.cluster.vq import kmeans, whiten
import matplotlib.pyplot as plt

# create 500 datapoints in two clusters a and b
pts = 1000
a = np.random.multivariate_normal([0,0],[[4,2],[2,4]], size = pts)
b = np.random.multivariate_normal([30,10],[[10,2],[2,1]], size = pts)
features = np.concatenate((a,b))

# whiten data
whitenedData = whiten(features)

# find 4 clusters in the data
points, d = kmeans(whiteneDada, 4)

# plot whitened data and cluster centers in red
plt.scatter(whitenedData[:,0], whitenedData[:,1])
plt.scatter(points[:,0], points[:,1], c = 'y')
plt.show()


# In[18]:


# Interpolate example
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
# np.linspace(start, stop, num)
x = np.linspace(0,4,10)
y = np.cos(x**2/3 + 4)
print(x)
print(y)
plt.scatter(x,y, c ='b')
plt.show()


# In[21]:


# Interpolate example
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Linear', 'Cubic'- kind represents the type of the interpolation technique. 
func1 = interp1d(x,y,kind = 'linear')
func2 = interp1d(x,y,kind = 'cubic')

# we difine a new set of input
x3 = np.linspace(0,4,100)
plt.plot(x,y,'o', x3, func1(x3), '-', x3, func2(x3), '--')
plt.legend(['data','linear', 'cubic','nearest'], loc = 'best')
# plt.show()


# In[31]:


# linear algebra example 1
import numpy as np
from scipy import linalg
#   2x + 4y - 6z = -6
#   2x - 5y + 4z = 13
#   10x + 8y - 2z = 10
# we will find the values of x, y and z for which all these equations are zero
# also we will check if the values are right by substituting them in the equations

# creating input array
aArr = np.array([[2,4,-6],[2,-5,4],[10,8,-2]])
# solution array
bArr = np.array([[-6], [13], [10]])

# Solve the problem using linear algebra
x = linalg.solve(aArr,bArr)
print(x)


# In[30]:


# Linear Algebra example 2
# Calculating determinant of a two-dimensional matrix
from scipy import linalg
import numpy as np
# define square matrix
arr = np.array([[4,5], [3,2]])
# pass values to det() function
linalg.det(arr)


# In[34]:


# Optimization and Fit example – scipy.optimize
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from scipy.optimize import minimize_scalar
k = np.arange(0, 1, 0.01)

def objective_function(x):
    return 4 * x ** 4 - 2 * x + 1 
plt.plot(k,objective_function(k))

res = minimize_scalar(objective_function)
print(res)


# In[37]:


import numpy as np
from scipy.optimize import minimize, LinearConstraint
def objective_func(x):
    return 2*x ** 4 - x ** 2
k = np.arange(-1, 1, 0.01)

plt.plot(k,objective_func(k))

ans = minimize_scalar(objective_function)
print(ans)


# In[39]:


ans2 = minimize_scalar(objective_function, method='bounded', bounds=(-1, 0))
print(res)


# In[40]:


# Image Processing example – scipy.ndimage
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
# get face image of panda from misc package
panda = misc.face()
# plot or show image of face
plt.imshow(panda)
# plt.show()


# In[42]:


# Cropping
lx,ly,channels = panda.shape
crop_panda = panda[int(lx/5):int(-lx/5), int(ly/5):int(-ly/5)]
plt.imshow(crop_panda)
# plt.show()


# In[43]:


# Rotation of Image using Scipy
from scipy import ndimage, misc
from matplotlib import pyplot as plt
panda = misc.face()
#rotatation function of scipy for image – image rotated 176 degree
panda_rotate = ndimage.rotate(panda, 176)
plt.imshow(panda_rotate)
# plt.show()


# In[45]:


# Blurring
from scipy import misc,ndimage
import matplotlib.pyplot as plt

panda = misc.face()
blurred = ndimage.gaussian_filter(panda, sigma=7)
plt.imshow(blurred)
# plt.show()


# In[47]:


# Special Function package example
# Cubic Root Function:
from scipy.special import cbrt
#Find cubic root of 27 & 64 using cbrt() function
cbRoot = cbrt([27, 64])
#print value of cb
print(cbRoot)


# In[76]:


# Exponential Function:
from scipy.special import exp10
# define exp10 function and pass value
# 10 to the power of x
exp = exp10([-1,0,1,10])
print(exp)


# In[72]:


# Combinations:
from scipy.special import comb
# Find combinations of 5, 2 values using comb(N, k)(expressed as “N choose k”)
combinationResult = comb(5, 2, exact = False, repetition=True)
print(combinationResult)


# In[48]:


# Permutations:
from scipy.special import perm
# Find permutation of 5, 2 using perm (N, k) function
permutationResult = perm(5, 2, exact = True)
print(permutationResult)


# In[ ]:




