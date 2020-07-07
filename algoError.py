#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Code for the algorithm's convergence with iterations and number of cost matrices
"""
import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import ot_max_funcs as otMax
from itertools import product
import cvxpy as cvx
from scipy.sparse import csr_matrix


#%%
m = 100    
n = 100
ones_s = np.ones(m)
ones_t = np.ones(n)
dimensions = range(10,100,10)
maxIter = 100
nRep = 30

errArray = np.zeros((nRep, len(dimensions), maxIter))
# supArray = np.zeros_like(errArray)
# infArray = np.zeros_like(errArray)

vis = True

if not vis: # True for visualization, false for running

    for i, j in product(range(nRep), range(len(dimensions))):
        d= dimensions[j]
        print("repetition = %d, dimension = %d"%(i, dimensions[j]))
        success = False
        while not success:
            try:

                X_s, y_s = make_blobs(n_features= d, n_samples= m)
                X_s = X_s / np.linalg.norm(X_s, axis=1, ord= np.inf)[:, None]
        
                X_t, y_t =  make_blobs(n_features= d, n_samples= n)
                X_t = X_t / np.linalg.norm(X_t, axis=1, ord= np.inf)[:, None]
        
                costMats = np.empty((d, m, n))
        
        
                for k in range(d):
                    X_s_pert = X_s + 0.1*np.random.uniform(-1, 1, size = X_s.shape)
                    X_t_pert = X_t + 0.1*np.random.uniform(-1, 1, size= X_t.shape)
                    costMats[k] = cdist(X_s_pert, X_t_pert)
        
        
                weightMats = csr_matrix(np.outer(ones_s/m, ones_t/n).reshape(1, m*n))
                G = np.array([np.einsum('ij, kij -> k', weightMats.toarray().reshape(m,n), costMats)])
                print("optimization starts ...")
                
                
                errList =    otMax.minimaxOT(r= np.ones(m)/m, c= np.ones(n)/n, maxIter= maxIter, threshold= 0, weightMats=weightMats, maxCost= otMax.maxCostFinite,updateFunc = otMax.updateFinite,
                                                arguments= (G, costMats), verbose= False, output= 2) 
                errArray[i,j, :len(errList)] = errList
                errArray[i,j, len(errList):] = errList[-1]
                success = True
            except cvx.SolverError:
                print("SOLVER ERROR CATCHED, RETRYING ... ")
                
                        

        #% save at each iteration
        np.save('errArray', errArray)
        # np.save('supArray', supArray)
        # np.save('infArray', infArray)
#%%
else:
#    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
#    rc('text', usetex=True)
    errArray = np.load('errArray.npy')
    # supArray   = np.load('supArray.npy')
    # infArray   = np.load('infArray.npy')

    def plotWithStd(xArray, yArray, axis, col, style, lw):
        meanYArray= np.median(yArray, axis=  axis).flatten()
        perc25 = np.quantile(yArray, q= 0.25, axis= axis).flatten()
        perc75 = np.quantile(yArray, q= 0.75, axis= axis).flatten()
        plt.semilogy(xArray, meanYArray, c = col, linestyle = style, lw = lw)
        plt.fill_between(xArray, perc25, perc75, facecolor= col, alpha= 0.1)
        return None

    plt.figure(figsize = (18,16))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    chosenDimInds = [0, 3, 8]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color'][:3]

    for i, color in zip(chosenDimInds, colors):#range(len(dimensions)):
        plotWithStd(range(maxIter), errArray[:,i,:], axis= 0, col = color, style = '-', lw = 8)
#    plotWithStd(dimensions, algoTime, axis= 0, col = 'black', style = '-.', lw = 8)
    plt.grid('on')
    plt.xlabel('iteration t', fontsize=40)
    plt.ylabel('err(t)', fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(["|C| = %d"%dimensions[i] for i in chosenDimInds],fontsize=40)
    plt.savefig('algoError.pdf', format='pdf', dpi= 1200)