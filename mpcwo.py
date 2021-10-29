# %%
import numpy as np
from cvxopt import solvers , matrix
import matplotlib.pyplot as plt
from combine import frames_to_video

st  = [1 , 1] 
en = [10  , 14 ]

n = 10 
dt = 0.1 
vmax = 30
Px = np.hstack(( np.ones((n , n)), np.zeros((n,n))))
Py = np.hstack(( np.zeros((n , n)), np.ones((n,n))))

P = 2* (dt**2 ) * np.vstack((Px, Py)) .astype(float)

q =  2*dt *np.array([st[0] - en[0]] * n  + [st[1] - en[1]] * n ).astype(float)
# print(q)
G = np.vstack( (np.diag([-1] * 2* n ) , np.diag([1]*2 *  n ))).astype(float)
h = np.array([0]* 2*n + [vmax ] * 2* n ).astype(float)
# print(G)
# print(h.shape , G.shape , P.shape , q.shape)
sol = solvers.qp(matrix(P) , matrix(q)  , matrix(G) , matrix(h))
s = np.array(sol['x']).ravel()
# print(sol['x'])
x = s[:10]*dt
y = s[10:]*dt 

x[0] += st[0]
y[0] += st[1]

x =[st[0]] + np.cumsum(x).tolist()
y =[st[1]] +  np.cumsum(y).tolist()

# print(x , y )

for i in range (2 , 1+ len(x)):
    plt.plot(x[i-2:i], y[i-2:i] , color="blue" )
    plt.scatter(x[:i],y[:i] ,color='red')
    plt.scatter([st[0] ,en[0]] , [st[1] , en[1]] , c='green')
    plt.savefig(f'mpc_without_o/{i}.png')
# print(sol['x'])
frames_to_video('./mpc_without_o/*.png' , './mpc_without_o/ta.mp4' , 5)
# %%
