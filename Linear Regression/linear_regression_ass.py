
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dfx = pd.read_csv('linearX.csv')
dfy = pd.read_csv('linearY.csv')
# Analyse the Data Frame
dfx.head(n=5)
#dfy.head(n=20)


# In[5]:


dfy.head(n=5)


# In[6]:


print(dfx.shape,dfy.shape)


# In[7]:


# Convert to Numpy Array 
# Values returns CSV to Numpy
x_values = dfx.values
target_values = dfy.values

plt.figure(0)
print(x_values.shape)
print(target_values.shape)


plt.title("Wines: Not normalized Data ")
plt.scatter(x_values,target_values,c='orange')
plt.xlabel("Acidity")
plt.ylabel("Density")
plt.show()


# In[8]:


mean = np.mean(x_values)
std_deviation = np.std(x_values)


x_values = x_values - mean
x_values /= std_deviation



print(mean)
print(std_deviation)

plt.figure(1)
plt.title("Wines Data - Normalized")
plt.xlabel("Acidity")
plt.ylabel("Density")
plt.scatter(x_values,target_values,c='orange')
plt.show()




# In[9]:


def getHypothesis(theta_val,x):
    x0 = 1
    bias = theta_val[0]*x0
    hx = theta_val[1]*x + bias 
    return hx
    
def getTotalError(target_value,input_x,theta):
    
    no_of_samples = input_x.shape[0]
    error = 0.0
    
    for i in range(no_of_samples):
        hxi =  getHypothesis(theta,input_x[i])
        small_error = (target_value[i] - hxi)**2
        error += small_error
        
    error /= 2
    
    return error

def getGradient(target_value,input_x,theta):
    
    
    no_of_samples = input_x.shape[0]
    
    grad = np.array([0.0,0.0]) 
    
    for i in range(no_of_samples):
        hxi = getHypothesis(theta,input_x[i])
        x0 = 1
        grad[0] += (target_value[i] - hxi)*x0
        grad[1] += (target_value[i]-hxi)*input_x[i]
    
    grad[0] /=no_of_samples
    grad[1] /=no_of_samples
    
    return grad
    
def gradientDescent(input_x,target_val,learning_rate=0.01,threshold_error=0.002,time_gap=0.2):
    
    theta = np.array([0.0,0.0])
    error = []
    
    e = getTotalError(target_val,input_x,theta)
    error.append(e)
    
    all_thetas = []
    
    start_time = time.clock()
    converged = False
    
    while(not converged):   
        
        grad = getGradient(target_val,input_x,theta)
        theta[0] = theta[0] + learning_rate*grad[0]
        theta[1] = theta[1] + learning_rate*grad[1]
        
        diff_time = time.clock() - start_time
        
        if(diff_time>=time_gap):
            start_time = time.clock()
            ne = getTotalError(target_val,input_x,theta)
            error.append(ne)
            
            if(e-ne<threshold_error):
                converged=True
            e = ne
            all_thetas.append((theta[0],theta[1]))
        
    
            
    return error,theta,all_thetas
      


# In[10]:


theta_init = np.array([0.0,0.0])
#print(getTotalError(target_values,x_values,theta))
#print(getGradient(target_values,x_values,theta))
err,theta,all_thetas = gradientDescent(x_values,target_values)

print(err[0],theta_init)
print(err[-1],theta)

plt.figure(2)
plt.title("Wines Data and Hypothesis Plot")
plt.xlabel("Acidity")
plt.ylabel("Density")
plt.scatter(x_values,target_values,c='orange',label="Data")

query_x = np.linspace(-2,5,10)
predicted_y = getHypothesis(theta,query_x)

plt.plot(query_x,predicted_y,label="Hypothesis")
plt.legend()
plt.show()






# In[11]:


plt.figure(2)
plt.ylabel("Squared Error")
plt.xlabel("Time")
plt.plot(err)
plt.title("Time Gap of 0.02s, Learning Rate : 0.01")
plt.show()




# In[22]:


from mpl_toolkits.mplot3d import Axes3D

# 3D Plot

def draw_3d_loss(all_thetas,err,contour=True):
    
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = "Contour Plot"
    X = np.arange(-4, 4, .02)
    Y = np.arange(-4, 4, .02)
    X, Y = np.meshgrid(X, Y)
    Z = np.ones(X.shape)
    m = X.shape[0] 
    n = X.shape[1]

    
    for i in range(m):
        for j in range(n):
            Z[i,j] = np.mean(0.5*(target_values - Y[i,j]*x_values -X[i,j])**2)

    
    
    
    cset = ax.contour(X, Y, Z, cmap="jet")
    ax.clabel(cset, fontsize=5, inline=1)
    
    plt.xlabel("Theta0")
    plt.ylabel("Theta1")
  
    
    for i in range(len(all_thetas)):
        ax.scatter(all_thetas[i][0],all_thetas[i][1],err[i],s=40,alpha=0.8, c='r',marker=r'^')
    
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel("Loss(J(Theta))")
    
    plt.title(title)
    plt.show()

draw_3d_loss(all_thetas,err)



# In[23]:


# Draw losses for various learning rates
rates = [0.001,0.005,0.009,0.013,0.017,0.021,0.025]

plt_id = 0
for lr in rates:
    
    err,theta,all_thetas = gradientDescent(x_values,target_values,lr)
    title = "Learning Rate : "+str(lr)
    draw_3d_loss(all_thetas,err,title)
    
    


# In[ ]:



# Try for Different Learning Rates
learning_rates = [ 0.001,0.005,0.009,0.013,0.017,0.021,0.025]
i=10

plot_id = 1
fig, axes = plt.subplots(nrows=4, ncols=2)
fig.tight_layout()
for lr in learning_rates:
    err,theta,all_thetas = gradientDescent(x_values,target_values,lr)
    plt.subplot(4,2,plot_id)
    plt.title("Learning Rate:"+str(lr))
    #plt.xlabel("Time")
    #plt.ylabel("Error")
    plt.plot(err)
    plot_id +=1 
    
plt.show()

    
    
    


# In[29]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




def meshPlot():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(-4, 4, .02)
    Y = np.arange(-4, 4, .02)
    X, Y = np.meshgrid(X, Y)
    m = X.shape[0] 
    n = X.shape[1]

    
    err,theta,all_thetas = gradientDescent(x_values,target_values,time_gap=0.02)

    Z = np.ones(X.shape)
    m = X.shape[0] 
    n = X.shape[1]

    for i in range(m):
        for j in range(n):
            Z[i,j] = np.mean(0.5*(target_values - Y[i,j]*x_values -X[i,j])**2)

        

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z,alpha=0.3, cmap="jet",antialiased=False)
    
    for i in range(len(all_thetas)):
        ax.scatter(all_thetas[i][0],all_thetas[i][1],err[i],s=40,alpha=1.0, c='k',marker=r'o')
    


    plt.xlabel("Theta0")
    plt.ylabel("Theta1")
    plt.title("Cost Vs Theta")
    #Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
meshPlot()

