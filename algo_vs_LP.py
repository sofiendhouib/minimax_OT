#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Code for the comparison of our algorithm vs Linear Programming
"""

import cvxpy as cvx
import numpy as np
from sklearn.datasets import make_blobs
from sys import path as path
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import ot_max_funcs as otMax
from time import time
from itertools import product
from scipy.sparse  import csr_matrix
from time import sleep



#%%

m = 100    
n = 100
ones_s = np.ones(m)
ones_t = np.ones(n)

dimensions = range(10, 110, 10)
nRep = 30
lpTime   = np.empty((len(dimensions), nRep))
lpValue = np.empty((len(dimensions), nRep))


algoTime = np.empty_like(lpTime)
algoValue = np.empty_like(lpTime)

vis = True # True for visualization, false for running

if not vis:

    for h, i in product(range(len(dimensions)), range(nRep)):
        d = dimensions[h]
        print("d= %d, repetition = %i" %(d,i))
        success= False
        while not success:
            try:
                X_s, y_s = make_blobs(n_features= d, n_samples= m)
                X_s = X_s / np.linalg.norm(X_s, axis=1, ord= np.inf)[:, None]
        
                X_t, y_t =  make_blobs(n_features= d, n_samples= n)
                X_t = X_t / np.linalg.norm(X_t, axis=1, ord= np.inf)[:, None]
        
                costMats = np.empty((d, m, n))
        
        
                for k in range(d):
                    X_s_pert = X_s + 0.1*(np.random.uniform(-1, 1, size = X_s.shape)-0.5)
                    X_t_pert = X_t + 0.1*(np.random.uniform(-1, 1, size= X_t.shape)-0.5)
                    costMats[k] = cdist(X_s_pert, X_t_pert)
        
                W = cvx.Variable((m,n))
                Maximum = cvx.Variable(1)
                constraints = [W >= 0, W*ones_t == ones_s/m, ones_s*W == ones_t/n]
                for k in range(d):
                    constraints.append(Maximum >= cvx.trace(W*costMats[k].T))
        
                problemLP = cvx.Problem(objective= cvx.Minimize(Maximum), constraints= constraints)
                sleep(1)
                startTime = time()
                problemLP.solve(solver= 'MOSEK', verbose= False, mosek_params = {'MSK_IPAR_INTPNT_MAX_ITERATIONS': int(1000),'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-8,
                                      'MSK_DPAR_DATA_TOL_X': 1e-8, 'MSK_DPAR_INTPNT_CO_TOL_REL_GAP':1e-8,
                                      'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8, 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8, "MSK_DPAR_BASIS_REL_TOL_S": 1e-8, 'MSK_DPAR_BASIS_TOL_S': 1e-8, 'MSK_DPAR_BASIS_TOL_X': 1e-8, "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-8,
                                      'MSK_IPAR_OPTIMIZER': 0})
                
#                problemLP.solve(solver= 'MOSEK', verbose= False, mosek_params = {'MSK_IPAR_INTPNT_MAX_ITERATIONS': int(1e7)})
                endTime = time()
                lpTime[h,i] = endTime - startTime
                print("LP time =%f" %lpTime[h,i])
                lpValue[h,i] = problemLP.value
                print("value at minimum: %f" %problemLP.value)
        
                weightMats = csr_matrix(np.outer(ones_s/m, ones_t/n).reshape(1, m*n))
                G = np.array([np.einsum('ij, kij -> k', weightMats.toarray().reshape(m,n), costMats)])
                
                sleep(1)
                startTime = time()
                W_algo, error, mu, C, arguments, dualVars, weightMats = otMax.minimaxOT(r= np.ones(m)/m, c= np.ones(n)/n, maxIter= int(1e3), threshold= 1e-8, weightMats= weightMats, maxCost= otMax.maxCostFinite,updateFunc = otMax.updateFinite,
                                                arguments= (G, costMats), verbose= False, output= 1)
                endTime = time()
                algoTime[h, i] = endTime - startTime
                print("algo time =%f" %algoTime[h,i])
                algoValue[h, i] = mu
                success = True
            except cvx.SolverError:
                print("SOLVER ERROR CATCHED, RETRYING ... ")
    plt.close('all')
    np.save('LP_time_1', lpTime)
    np.save('LP_value_1', lpValue)
    np.save('RPK_time_1', algoTime)
    np.save('RPK_value_1', algoValue)
#%%
else:
    lpTime = np.load('LP_time_1.npy')
    lpValue = np.load('LP_value_1.npy')
    algoTime = np.load('RPK_time_1.npy')
    algoValue = np.load('RPK_value_1.npy')

    def plotWithStd(xArray, yArray, axis, col, style, lw):
        meanYArray= np.median(yArray, axis=  axis).flatten()
        # stdYArray= np.std(yArray, axis= axis).flatten()
        perc25= np.quantile(yArray, q= 0.25, axis= axis).flatten()
        perc75= np.quantile(yArray, q= 0.75, axis= axis).flatten()
        plt.plot(xArray, meanYArray, c = col, linestyle = style, lw = lw)
        plt.fill_between(xArray, perc25, perc75, facecolor= col, alpha=0.1)
        # plt.fill_between(xArray, meanYArray - stdYArray, meanYArray + stdYArray, alpha=0.1, facecolor=col)
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
    colors = prop_cycle.by_key()['color'][:2]
    plotWithStd(dimensions, lpTime, axis= 1, col = colors[0], style = '-', lw = 8)
    plotWithStd(dimensions, algoTime, axis= 1, col = colors[1], style = '-.', lw = 8)
    plt.grid('on')
    plt.xlabel('Number of cost matrices', fontsize=40)
    plt.ylabel('Time(s)', fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(["LP", "RKP"],fontsize=40)
    plt.savefig('LPvsRPK-time1.pdf', format='pdf', dpi=1200)
# =============================================================================
#     
# =============================================================================
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
    colors = prop_cycle.by_key()['color'][:2]
#    plotWithStd(dimensions, lpValue, axis= 1, col = colors[0], style = '-', lw = 8)
#    plotWithStd(dimensions, algoValue, axis= 1, col = colors[1], style = '-.', lw = 8)
    plotWithStd(dimensions, (algoValue - lpValue)/lpValue, axis= 1, col = colors[0], style = '-', lw = 8)
    plt.grid('on')
    plt.xlabel('Number of cost matrices', fontsize=40)
    plt.ylabel('algorithm - LP value', fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(["LP", "RKP"],fontsize=40)
    plt.savefig('LPvsRPK-value.pdf', format='pdf', dpi=1200)