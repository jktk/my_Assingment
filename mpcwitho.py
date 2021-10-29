#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from cvxopt import solvers , matrix
import matplotlib.pyplot as plt
from combine import frames_to_video

# In[2]:


#Inputs
st = [1, 1] #Starting index
en = [14, 15] #Ending point
obs = [7,8] #Location of the object

n = 15  #Max number of steps
dt = 0.1 #Time for the step 
vmax = 30 #Max velocity along a direction
Rr =1 #Radius of robot
Ro =2 #Radius of circular obstacle`



# In[3]:


# Objective function
Px = np.hstack(( np.ones((n , n)), np.zeros((n,n))))
Py = np.hstack(( np.zeros((n , n)), np.ones((n,n))))

P = (2* (dt**2 ) * np.vstack((Px, Py))).astype(float)

q = 2*dt*np.array([st[0] - en[0]]*n  + [st[1] - en[1]]*n).astype(float)


# In[4]:


#Speed constraint
G = np.vstack( (np.diag([-1] * 2* n ) , np.diag([1]*2 *  n ))).astype(float)
h = np.array([0]* 2*n + [vmax ] * 2* n ).astype(float)

# In[5]:

# Obstacle constraint
c1=(st[0]-obs[0])**2
c2=(st[1]-obs[1])**2
R=(Rr+Ro)**2
xst = np.random.uniform(3, vmax/2,n)
yst = np.random.uniform(3, vmax/2,n)

t = 5
Gtemp = G 
htemp = h

for _ in range(t):
    Gtemp = G 
    htemp = h
    for i in range(1, n +1): 
        v = np.array([2*dt * (st[0] - obs[0]  +  dt* np.sum(xst[:i-1]))]*i + [0]*(n-i) + [2*dt * (st[1] - obs[1]  +  dt *np.sum(yst[:i-1]))]*i + [0]*(n-i))
        # print(v)
        Gtemp = np.vstack((Gtemp , -v ))
        htemp = np.insert(htemp , len(htemp) , c1 + c2 - R - (np.sum(xst[:i-1])*dt)**2 - (np.sum(yst[:i-1])*dt)**2)
    sol = solvers.qp(matrix(P) , matrix(q)  , matrix(Gtemp) , matrix(htemp))
    s = np.array(sol['x']).ravel()
    xs =np.array( s[:n])
    ys =np.array( s[n:])


# print(h)

#Solving quadratic programming using cvxopt
sol = solvers.qp(matrix(P) , matrix(q)  , matrix(Gtemp) , matrix(htemp))
# It doesn't have any equal to constraint

# Getting the values of velocities
s = np.array(sol['x']).ravel()
# print(sol['x'])
x = s[:n]*dt
y = s[n:]*dt 

x[0] += st[0]
y[0] += st[1]

x =[st[0]] + np.cumsum(x).tolist()
y =[st[1]] +  np.cumsum(y).tolist()
# In[7]:

fig, ax = plt.subplots()
# plt.plot(x, y )
# # print(sol['x'])
# # plt.scatter([obs[0]],[obs[1]], c='blue')
# ax = ax.add_artist(  plt.Circle((obs[0],obs[1]),Rr + Ro,color='blue'))
# plt.scatter(x,y, c='red')
# plt.scatter([en[0]] , [en[1]] , c='green')
# plt.show()
ax = ax.add_artist(  plt.Circle((obs[0],obs[1]),Rr + Ro,color='blue'))

for i in range (2 , 1+ len(x)):
    plt.plot(x[i-2:i], y[i-2:i] , color="blue" )
    plt.scatter(x[:i],y[:i] ,color='red')
    plt.scatter([st[0] ,en[0]] , [st[1] , en[1]] , c='green')
    plt.savefig(f'mpc_with_o/{i}.png')
# print(sol['x'])
frames_to_video('./mpc_with_o/*.png' , './mpc_with_o/out.mp4' , 1)
