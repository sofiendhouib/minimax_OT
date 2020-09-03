#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Code for the fragmented hypercube experiment
    Uses functions from the Subspace Robust Wasserstein Distances' code:
        https://github.com/francoispierrepaty/SubspaceRobustWasserstein
"""
import numpy as np
import ot_max_funcs as otMax
from matplotlib import pyplot as plt
from ot import emd, sinkhorn
from scipy.spatial.distance import cdist
import cvxpy as cvx
from itertools import combinations
from sys import path
path.append("./SubspaceRobustWasserstein/")
from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe
from scipy.sparse import csr_matrix, vstack

plt.rcParams.update({"font.family":"serif",
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         r"\usepackage{amsmath, amsfonts, amssymb, amstext, amsthm, bbm, mathtools}",
         ]
})
plt.rc('text', usetex=True)

plt.close('all')
#%%
def T(x,d,dim=2, pos= 0, magnitude= 1):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)
    return x + 2*magnitude*np.sign(x)*np.array([0]*pos + dim*[1]+(d-dim-pos)*[0])

def fragmented_hypercube(n, d, dim, pos, magnitude= 1):
    assert dim <= d
    assert dim >= 1
    assert dim == int(dim)

    a = (1. / n) * np.ones(n)
    b = (1. / n) * np.ones(n)

    # First measure : uniform on the hypercube
    X = np.random.uniform(-1, 1, size=(n, d))

    # Second measure : fragmentation
    Y = T(np.random.uniform(-1, 1, size=(n, d)), d, dim, pos= pos, magnitude= magnitude)

    return a, b, X, Y

d = 30
n = 250
dims = [2, 4, 7, 10]
k_star = 2

a,b,cube,transformCube = fragmented_hypercube(n,d,k_star, pos= 0, magnitude= 1)

ones = np.ones(n)

#transformCube= cube + 2*np.sign(cube)*np.sum(np.eye(d)[:k_star], axis= 0) # fragmented cube
#transformCube = np.random.permutation(transformCube)


#weightMats = [emd(ones/n, ones/n, cdist(cube, transformCube, 'sqeuclidean'))] # Wasserstein transport 
weightMats = [np.outer(ones/n, ones/n)] # independent distributions transport 


# %% Mahalanobis with SVM solver
# weightMats = csr_matrix(np.outer(ones/n, ones/n).reshape(1, n*n))
# stBox = np.einsum('ij, ik -> ijk', otMax.differences(cube, transformCube), otMax.differences(cube, transformCube)).reshape((n*n, d*d))
# data = np.dot(weightMats.toarray(), stBox)
# data = np.vstack((data, -data))
# y_data = np.array([1,-1])
# #normStBox = np.linalg.norm(stBox, ord= 2)
# weightMats = vstack((weightMats,-weightMats))
# #W_Mahalanobis, error, mu = otMax.minimaxOT(ones/n, ones/n, maxIter= 200, threshold= 1e-8, weightMats= weightMats, maxCost= otMax.maxCostMahalanobisSVM, updateFunc= otMax.updateMahalanobisSVM,
# #                                  arguments= (data, y_data, stBox, d, n, n), verbose= True)#%%

# W_Mahalanobis, error, muValueSVM, C_star, arguments, dualVars, weightMats = otMax.minimaxOT(ones/n, ones/n, maxIter= 200, threshold= 1e-8, weightMats= weightMats, maxCost= otMax.maxCostMahalanobisSVM, updateFunc= otMax.updateMahalanobisSVM,
#                                   arguments= (data, y_data, stBox, d, n, n), verbose= True, output= 1)#%%


#%% Mahalanobis dual formùlation
W0 = emd(ones/n, ones/n, cdist(cube, transformCube))
weightMats = csr_matrix(W0.reshape(1, n*n))
V0 = otMax.vecDisplacementMatrix(cube, transformCube, W0)
V0 = V0.reshape(1,len(V0))
W_Mahalanobis, error, muValueDual, C_star, arguments, dualVars, weightMats = otMax.minimaxOT(ones/n, ones/n, maxIter= 400, threshold= 1e-8, 
                                                                                             weightMats= weightMats, 
                                                                                             maxCost= otMax.maxCostMahalanobisDual, 
                                                                                             updateFunc= otMax.updateMahalanobisDual, 
                                                                                             arguments= (V0, cube, transformCube, None), 
                                                                                             verbose= False, output= 1)

#%% finite number of 1D projections
costMats = np.empty((d, n, n))
for k in range(d):
    costMats[k] = np.abs(cube[:,k][:,None] - transformCube[:,k])**2
weightMats = csr_matrix(np.outer(ones/n, ones/n).reshape(1, n*n))
G = np.array([np.einsum('ij, kij -> k', weightMats.toarray().reshape(n,n), costMats)])

W_finite_1, error, muValueDual, C_star, arguments, dualVars, weightMats = otMax.minimaxOT(r= ones/n, c= ones/n, maxIter= 100, threshold= 1e-9, weightMats= weightMats, maxCost= otMax.maxCostFinite,updateFunc = otMax.updateFinite,
                        arguments= (G, costMats), verbose= False, output= 1)
#%% finite number of 2D projections 
costMats = []
tupleList = []
for t in combinations(range(d),k_star):
    tupleList.append(t)
    costMats.append(cdist(cube[:,[*t]], transformCube[:, [*t]], "sqeuclidean"))

costMats = np.array(costMats)
weightMats = csr_matrix(np.outer(ones/n, ones/n).reshape(1, n*n))
G = np.array([np.einsum('ij, kij -> k', weightMats.toarray().reshape(n,n), costMats)])

W_finite_2, error, muValueDual, C_star, arguments, dualVars, weightMats = otMax.minimaxOT(r= ones/n, c= ones/n, maxIter= 100, threshold= 1e-9, weightMats= weightMats, maxCost= otMax.maxCostFinite,updateFunc = otMax.updateFinite,
                        arguments= (G, costMats), verbose= False, output= 1)

#%% SRW code
FW = FrankWolfe(reg=1, step_size_0=None, max_iter=1000, threshold=0.01, max_iter_sinkhorn=50, threshold_sinkhorn=1e-4, use_gpu=False)
SRW = SubspaceRobustWasserstein(cube, transformCube, ones/n, ones/n, FW, 2)
SRW.run()
W_SRW = SRW.pi

#%% Plotting the transport
# For Wasserstein distance
W_Wasserstein = emd(ones/n, ones/n, cdist(cube, transformCube, 'sqeuclidean'))

W_list    = [
        W_Wasserstein, 
            W_Mahalanobis, 
#            W_sinkh,
            W_finite_1,
            W_finite_2, 
            W_SRW,
            ]
title_list = [
              "Wasserstein", 
              "RKP-Mahalanobis", 
              "RKP-1D-projections",
              "RKP-2D-projections", 
               "SRW",
              ]



# print("plotting ...") 
plt.close('all')
for W , title in zip(W_list, title_list):
    otMax.plotTransport(cube, transformCube, W, title= None)
    plt.axis("off")
    plt.savefig("./hypercube/"+title + ".pdf", format= "pdf", bbox_inches='tight', pad_inches= 0) 
