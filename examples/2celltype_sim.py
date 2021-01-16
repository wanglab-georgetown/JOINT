# This code shows how to use JOINT to perform soft clustering and do DEG

import numpy as np
from joint import joint
from joint import deg


# generate mixture of zero inflated negative binomial random variables
np.random.seed(12345)

K = 2 # number of cell types
G = 2 # number of genes
L = 2 # number of negative binomial components + 1

b = 1.0
A = [[5,10],[20,30]]
P = []
for a in A:
    t = []
    for x in a:
        t.append([x,b])
    P.append(t)
P = np.array(P)

ks = np.array(range(K))
pi = np.array([0.4,0.6])

th = 0.2

C = 5000 # number of cells
xs = []
ck = []
for i in range(C):
    ys = []
    k=np.random.choice(ks,p=pi)
    ck.append(k)
    for n in range(G):
        la = np.random.gamma(P[k,n,0], P[k,n,1])
        s = np.random.poisson(la)
        t = np.random.rand()
        y = s*(t>th)
        ys.append(y)
    xs.append(ys)
x0 = np.array(xs).T
ck = np.array(ck)
labels = ck

# perform soft clustering EM algorithm
sol_joint = joint.joint(x0, K, L, n_inits=2, n_init_iter=1,normalize_data=False)

# DEG with unknown labels
# perform DEG using the clustering results
sol_deg0 = deg.deg_unknown_labels(x0, K, 0,1,em_res=sol_joint,n_inits=2, n_init_iter=1,normalize_data=False)

# perform DEG directly without using the clustering results
sol_deg1 = deg.deg_unknown_labels(x0, K, 0,1,n_inits=2, n_init_iter=1,normalize_data=False)

# DEG with known labels
sol_deg2 = deg.deg_known_labels(x0, K, 0,1,labels=labels,n_inits=2, n_init_iter=1,normalize_data=False)


print(sol_joint,sol_deg0,sol_deg1,sol_deg2)
