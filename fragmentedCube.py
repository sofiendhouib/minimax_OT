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
path.append("../../SubspaceRobustWasserstein-master/")
from SRW import SubspaceRobustWasserstein
from Optimization.frankwolfe import FrankWolfe
from scipy.sparse import csr_matrix, vstack
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

d = 50
n = 100
dims = [2, 4, 7, 10]
k_star = 10

a,b,cube,transformCube = fragmented_hypercube(n,d,k_star, pos= 0, magnitude= 0.5)

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
W_Mahalanobis, error, muValueDual, C_star, arguments, dualVars, weightMats = otMax.minimaxOT(ones/n, ones/n, maxIter= 400, threshold= 1e-10, 
                                                                                             weightMats= weightMats, 
                                                                                             maxCost= otMax.maxCostMahalanobisDual, 
                                                                                             updateFunc= otMax.updateMahalanobisDual, 
                                                                                             arguments= (V0, cube, transformCube, None), 
                                                                                             verbose= True, output= 1)
V = arguments[0][:-1,:]

#%
M0 = V0.reshape((d,d))
eigValsM0 = np.linalg.eigvalsh(M0)
diagM0 = np.diag(M0)

M = np.dot(dualVars, V).reshape((d,d)) 
diagM = np.diag(M)
eigValsM = np.linalg.eigvalsh(M)
#%%
plt.figure()
plt.plot(np.arange(1,d+1), eigValsM0[::-1]/max(eigValsM0))
plt.plot(np.arange(1,d+1), eigValsM[::-1]/max(eigValsM))
plt.legend(['EMD', 'minimax Mahalanobis EMD'])
plt.title('Spectrum of second order displacement matrix, $k^* = %d$, normalized so max = 1'%k_star)
plt.plot([k_star, k_star], [0,1], '--')

#%%
plt.figure()
plt.plot(np.arange(1,d+1), diagM0/max(diagM0))
plt.plot(np.arange(1,d+1), diagM/max(diagM))
plt.legend(['EMD', 'minimax Mahalanobis EMD'])
plt.title('Diagonal of second order displacement matrix, $k^* = %d$, normalized so max = 1'%k_star)
plt.plot([k_star, k_star], [0,1], '--')

#%% Sinkhorn 
#W = np.outer(ones/n, ones/n)
#error = 1e16
#errorList = []
#p = 2
#if p == 1: 
#    q = np.inf
#    r = np.inf
#elif p == np.inf: 
#    q = 1
#    r = 0
#else: 
#    q = p/(p-1)
#    r  = 1/(p-1)
#i = 0
#while error >1e-10:
#    M = otMax.vecDisplacementMatrix(cube, transformCube, W).reshape((d,d))
#    eigVals, P = np.linalg.eigh(M)
#    normM1 = np.linalg.norm(eigVals, ord= p)
#    normM2 = np.linalg.norm(eigVals, ord= q)
#    recEigVals = (eigVals/normM2)**r
#    recM = np.linalg.multi_dot((P, np.diag(recEigVals), P.T))
#    C = otMax.sqMahalanobisMat(cube, transformCube, recM, cvxpy= False)
#    W = sinkhorn(ones/n, ones/n, C/np.max(C),reg= 0.1, numItermax= 100000, stopThr= 1e-6)
#    error = abs(normM2 - np.sum(W*C))
#    errorList.append(error)
#    print(error)
#    i+= 1
#
#W_sinkh = W.copy()
#plt.figure()
#plt.plot(np.arange(1,d+1), eigVals)
#plt.plot(np.arange(1,d+1), recEigVals)
#%% Mahalanobis with CVXPY
# M = cvx.Variable((d,d))
# sqMahalanobis = otMax.sqMahalanobisMat(cube, transformCube, M, cvxpy= True)
# constraints = [cvx.trace(W.toarray().reshape((n,n)).T*sqMahalanobis) >= 1 for W in weightMats]

# W_Mahalanobis, error, muValuePrimal = otMax.minimaxOT(ones/n, ones/n, maxIter= 100, threshold= 1e-5, weightMats= weightMats, maxCost= otMax.maxCostMahalanobis, updateFunc= otMax.updateMahalanobis,
#                                   arguments= (M, sqMahalanobis, constraints, cube, transformCube), verbose= True)

#%%
# print(muValueSVM - muValueDual)

#%% low rank formulation (same as in Cuturi's paper)
#n = cvx.Variable((d,d), PSD= True)
#rank = k_star
#weightMats = [np.outer(ones/n, ones/n)]
#constraints  = [n << cvx.trace(n)*np.eye(d)/rank]
#sqMahalanobis = otMax.sqMahalanobisMat(cube, transformCube, n, cvxpy= True)
#constraints += [cvx.trace(W.T*sqMahalanobis) >= 1 for W in weightMats]
#W, error = otMax.minimaxOT(ones/n, ones/n, maxIter= 100, threshold= 1e-9, maxCost= otMax.maxCostLowRank, updateFunc= otMax.updateLowRank,
#                                  arguments= (n, sqMahalanobis, constraints, cube, transformCube, rank), verbose= True)

#%%
#costMats = np.empty((d, n, n))
#angles = np.linspace(0,100,101)
#for k in range(d):
##for angle in angles:
##    theta = np.deg2rad(angle)
##    vec = np.dot(np.array([1,0]), np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]))
##    costMats.append(np.abs(np.dot(cube,vec)[:,None] - np.dot(transformCube,vec))**2)#[k] = np.abs(cube[:,k][:,None] - transformCube[:,k])**2
#    costMats[k] = np.abs(cube[:,k][:,None] - transformCube[:,k])**2
#weightMats = csr_matrix(np.outer(ones/n, ones/n).reshape(1, n*n))
#G = np.array([np.einsum('ij, kij -> k', weightMats.toarray().reshape(n,n), costMats)])
#
#W_finite_1, error,_, C_star, _ = otMax.minimaxOT(r= ones/n, c= ones/n, maxIter= 100, threshold= 1e-9, weightMats= weightMats, maxCost= otMax.maxCostFinite,updateFunc = otMax.updateFinite,
#                        arguments= (G, costMats), verbose= False, output= 1)

#%%
#costMats = []
#tupleList = []
#for t in combinations(range(d),k_star):
#    tupleList.append(t)
#    costMats.append(cdist(cube[:,[*t]], transformCube[:, [*t]], "sqeuclidean"))
#
#costMats = np.array(costMats)
#weightMats = csr_matrix(np.outer(ones/n, ones/n).reshape(1, n*n))
#G = np.array([np.einsum('ij, kij -> k', weightMats.toarray().reshape(n,n), costMats)])
#
#W_finite_2, error,_, C_star, arguments = otMax.minimaxOT(r= ones/n, c= ones/n, maxIter= 100, threshold= 1e-9, weightMats= weightMats, maxCost= otMax.maxCostFinite,updateFunc = otMax.updateFinite,
#                        arguments= (G, costMats), verbose= False, output= 1)

#%% Cuturi's code
#FW = FrankWolfe(reg=1, step_size_0=None, max_iter=1000, threshold=0.01, max_iter_sinkhorn=50, threshold_sinkhorn=1e-4, use_gpu=False)
#SRW = SubspaceRobustWasserstein(cube, transformCube, ones/n, ones/n, FW, 2)
#SRW.run()
#W_SRW = SRW.pi

#%% Plotting the transport
# For Wasserstein distance
W_Wasserstein = emd(ones/n, ones/n, cdist(cube, transformCube, 'sqeuclidean'))

W_list    = [
        W_Wasserstein, 
            W_Mahalanobis, 
#            W_sinkh,
#             W_finite_1,
#             W_finite_2, 
#             W_SRW,
            ]
title_list = [
              "Wasserstein", 
#              "RPK: Mahalanobis", 
              "RPK: Mahalanobis+sinkhorn"
              # "RPK: 1D projections", e
              # "RPK: 2D projections", 
#              "SRW"
              ]


eigVals = np.linalg.eigvalsh(M)  

#assert d-np.argmax(np.diff(eigVals))-1 == k_star

plt.figure()
plt.imshow(M)
plt.title("k^* = %d" %k_star)
plt.savefig("M_k=%d.png"%k_star)

#plt.figure()
#plt.imshow(recM)
#plt.title("k^* = %d" %k_star)
#plt.savefig("M_k=%d.png"%k_star)
# print("plotting ...") 
# for W , title in zip(W_list, title_list):
#     otMax.plotTransport(cube, transformCube, W, title)
#     plt.savefig(title + ".png")
