#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Code for the algorithm's convergence with iterations and regularization
"""
import numpy as np
from sklearn.datasets import make_blobs
from sys import path as path
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import ot_max_funcs as otMax
from itertools import product
import cvxpy as cvx
from scipy.sparse import csr_matrix
from time import time


#%%
m = 100    
n = 100
d= 40

ones_s = np.ones(m)
ones_t = np.ones(n)

regList = [1, 0.1, 0.01]
maxIter = 100
nRep = 30

errArray = np.zeros((nRep, len(regList), maxIter))
supArray = np.zeros_like(errArray)
infArray = np.zeros_like(errArray)

vis =  True# True for visualization, false for running
if not vis:

    for i, j in product(range(nRep), range(len(regList))):
        reg= regList[j]
        print("repetition = %d, reg= %e"%(i,reg))
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
                
                
                W0 = np.outer(ones_s/m, ones_t/n)
                weightMats = W0.reshape(1, m*n)
                G = np.array([np.einsum('ij, kij -> k', W0, costMats)])
                startTime = time()
                errList = otMax.minimaxOT(r= np.ones(m)/m, c= np.ones(n)/n, maxIter= maxIter, threshold= 0, weightMats= weightMats, 
                                                     maxCost= otMax.maxCostFiniteReg,updateFunc = otMax.updateFiniteReg,
                                                arguments= (G, costMats), verbose= False, output= 2, reg=reg)
                errArray[i,j, :len(errList)] = errList
                errArray[i,j,len(errList):] = errList[-1]
                success = True
            except cvx.SolverError:
                print("SOLVER ERROR CATCHED, RETRYING ... ")
                
                        

        #% save at each iteration
        np.save('errArraySinkhorn', errArray)
        # np.save('supArraySinkhorn1', supArray)
        # np.save('infArraySinkhorn1', infArray)
#%%
else:
#    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
#    rc('text', usetex=True)
    errArray = np.load('errArraySinkhorn.npy')
    # supArray   = np.load('supArraySinkhorn1.npy')
    # infArray   = np.load('infArraySinkhorn1.npy')

    def plotWithStd(xArray, yArray, axis, col, style, lw):
        meanYArray= np.median(yArray, axis=  axis).flatten()
#        stdYArray= np.std(yArray, axis= axis)
        perc25 = np.quantile(yArray, q= 0.25, axis= axis).flatten()
        perc75 = np.quantile(yArray, q= 0.75, axis= axis).flatten()
        perc10 = np.quantile(yArray, q= 0.1, axis= axis).flatten()
        perc90 = np.quantile(yArray, q= 0.9, axis= axis).flatten()
        plt.semilogy(xArray, meanYArray, c = col, linestyle = style, lw = lw)
#        plt.fill_between(xArray, meanYArray - stdYArray, meanYArray + stdYArray, alpha=0.1, facecolor=col)
        plt.fill_between(xArray, perc25, perc75, facecolor= col, alpha= 0.1)
#        plt.fill_between(xArray, perc10, perc90, facecolor= col, alpha= 0.05)
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
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i, color in zip(range(len(regList)), colors):#range(len(regList)):
        plotWithStd(range(maxIter), errArray[:,i,:], axis= 0, col = color, style = '-', lw = 8)
#    plotWithStd(regList, algoTime, axis= 0, col = 'black', style = '-.', lw = 8)
    plt.grid('on')
    plt.xlabel('iteration t', fontsize=40)
    plt.ylabel('err(t)', fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend([r"$\lambda$ = %.1e"%regList[i] for i in range(len(regList))],fontsize=40)
    plt.savefig('algoErrorSinkhorn.pdf', format='pdf', dpi= 1200)