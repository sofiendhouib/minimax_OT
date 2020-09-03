#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Implementation of the our robust optimal transport formulation
"""
from ot import emd
from ot import sinkhorn
from ot.bregman import sinkhorn_epsilon_scaling
import cvxpy as cvx
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
# from myDA_funcs import kp
from sklearn.metrics.pairwise import manhattan_distances
from cvxopt import solvers

solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')  # to silence GLPK outputs
from scipy.sparse import csr_matrix, vstack, issparse

"""
    Implementation of the minimax OT algorithm using the cutting plane method described in:
        * [Blankenship and Falk, 1976]: https://link.springer.com/article/10.1007/BF00934096
        * [Mutapcic and Boyd, 2009]: https://web.stanford.edu/~boyd/papers/prac_robust.html
    
    
    Here, the pessimizing step is the transport computation at each step of the algorithm.
    The current version implements non-binding constraints elimination, implemented in the first reference
"""

# %%
"""
    Main function to run the minimax algorithm for optimal transport
    arguments:
        * maxCost: function that solves the maximum cost problem (max mu in OTML paper)
        * updateFunc: function that updates the problem at the end of each iteration
        * arguments: arguments of the maxCost function, given as a tuple
"""


def minimaxOT(r, c, maxIter, threshold, weightMats, maxCost, updateFunc, arguments, verbose=False, output=0, reg=0):
    error = 1
    if output == 2:
        errorList = []
        supList = []
        infList = []
    converged = False
    muPrev = 1e16
    for i in range(maxIter):
        print(".", end="")
        newW = np.zeros((len(r), len(c)))

        if output == 3:
            muValue, C_star, dualVars, M_plot = maxCost(*arguments, dim_M=True)
        else:
            muValue, C_star, dualVars = maxCost(*arguments)

        #        if muValue > oldMu: break
        if C_star is None:
            return newW, error

        if reg == 0:
            newW = emd(r, c, C_star, numItermax=10000000)
            newW[newW < 1e-12] = 0
            newW = csr_matrix(newW)
            error = abs(muValue - np.sum(newW.multiply(C_star)))
        else:
            try:
                newW = sinkhorn(r, c, C_star, reg)
                error = abs(muValue - np.einsum('ij,ij', newW, C_star))
            except ValueError:
                print("q")

        if verbose:
            print("\n")
            print("iteration = %d" % i)
            print("error = %e" % error)
            print("nb of transport matrices: %d" % len(dualVars))

        if error <= threshold or abs(muValue - muPrev) <= threshold ** 2:
            converged = True
            print("\\", end="")
            break

        if output == 2:
            errorList.append(error)
            # supList.append(muValue)
            # infList.append(np.einsum('ij,ij', newW.toarray(), C_star))

        #            print("mu = %f"%muValue)
        updateOutput = tuple(updateFunc(newW, dualVars, weightMats, *arguments))
        arguments = updateOutput[2:]
        dualVars = updateOutput[0]
        weightMats = updateOutput[1]
        muPrev = muValue
    if output == 2:
        return errorList
    # else:
    #     if not converged: weightMats = weightMats[:-1, :]     

    # if not converged: weightMats = weightMats[:-1, :]     

    # solving the dua to get W that minimizes max_C <W, C>
    # toKeep = dualVars>0
    #    dualVars = dualVars[toKeep]
    #    weightMats = np.abs(weightMats[toKeep])
    # weightMats = weightMats[:-1,:]
    # W = weightMats.T.dot(dualVars).reshape(newW.shape)
    # W /= np.sum(np.abs(dualVars))
    # W = csr_matrix(W)

    # if  output == 0:
    #     return W, error, muValue

    else:
        # solving the dua to get W that minimizes max_C <W, C>
        toKeep = dualVars > 1e-12
        dualVars = dualVars[toKeep]
        weightMats = np.abs(weightMats[toKeep])
        W = weightMats.T.dot(dualVars).reshape(newW.shape)
        W /= np.sum(np.abs(dualVars))
        W[W < 1e-12] = 0
        W = csr_matrix(W)

        if output == 0:
            return W, error, muValue

        elif output == 1:
            return W, error, muValue, C_star, arguments, dualVars, weightMats

        elif output == 3:
            return W, error, muValue, C_star, M_plot


def miniminimaxOT(r, c, maxIter, threshold, maxCost, updateFunc, Xt, Xs, C_center=None, verbose=False,
                  reg=[0], uniform_init=True, ball_radius=0.001, normalize="OT", output=1):
    if C_center is None:
        C_center = np.zeros((1, len(r), len(c)))

    n = len(r)
    m = len(c)

    best_T = np.empty_like(C_center)
    best_C = np.empty_like(C_center)

    norm_term_list = []
    Tc_list = []
    M_mahalanobis = []
    for c_index, c_center in enumerate(C_center):
        print("\n", c_index, end="")
        if len(reg) == len(C_center):
            reg_ = reg[c_index]
        else:
            reg_ = reg[0]

        # Normalize by the OT on c_center
        if normalize == "OT":
            if reg_ == 0:
                Tc = emd(r, c, c_center)
            else:
                Tc = sinkhorn_epsilon_scaling(r, c, c_center, reg=reg_)
            norm_term = np.sum(Tc * c_center)
        elif normalize == "C_norm":
            norm_term = np.linalg.norm(c_center)
            Tc = None
        elif normalize == "max":
            norm_term = np.max(c_center)
            Tc = None

        c_center = c_center / norm_term
        reg_ = reg_ / norm_term

        norm_term_list.append(norm_term)
        Tc_list.append(Tc)

        if type(Xt) is list:
            Xt_ = Xt[c_index]
            Xs_ = Xs[c_index]
        else:
            Xt_ = Xt
            Xs_ = Xs

        d = Xt_.shape[1]

        M = cvx.Variable((d, d), symmetric=True)

        # Initialization of the Transport map
        if uniform_init:
            weightMats = np.outer(np.ones(n) / n, np.ones(m) / m)
        else:
            weightMats = Tc

        # Compute the mahalanobis centered on 0.
        sqMahalanobis = sqMahalanobisMat(Xs_, Xt_, M, cvxpy=True)

        # mu variable
        mu = cvx.Variable(1)

        # Compute the constrainte
        constraints = [cvx.sum(M ** 2) <= ball_radius]
        constraints += [cvx.trace(weightMats.T * (sqMahalanobis + c_center)) >= mu]
        weightMats = csr_matrix(weightMats.reshape(1, n * m))
        if output == 3:
            W, error, muValue, C_star, M_i = minimaxOT(r, c, maxIter=maxIter,
                                                       threshold=threshold,
                                                       maxCost=maxCost,
                                                       updateFunc=updateFunc,
                                                       arguments=(
                                                           M, mu, sqMahalanobis, constraints, Xs_, Xt_, c_center),
                                                       verbose=verbose,
                                                       reg=reg_,
                                                       weightMats=weightMats,
                                                       output=output)
            M_mahalanobis.append(M_i)

        else:
            W, error, muValue, C_star, arguments, dualVars, weightMats = minimaxOT(r, c, maxIter=maxIter,
                                                                                   threshold=threshold,
                                                                                   maxCost=maxCost,
                                                                                   updateFunc=updateFunc,
                                                                                   arguments=(
                                                                                       M, mu, sqMahalanobis,
                                                                                       constraints, Xs_, Xt_, c_center),
                                                                                   verbose=verbose,
                                                                                   reg=reg_,
                                                                                   weightMats=weightMats,
                                                                                   output=output)

        best_T[c_index] = W.toarray().reshape(len(r), len(c))
        best_C[c_index] = C_star

    best_C_position = np.argmin((best_C * best_T).sum(axis=(1, 2)))
    if output == 3:
        return C_center[best_C_position], best_T, best_C, np.array(norm_term_list), Tc_list, M_mahalanobis
    else:
        return C_center[best_C_position], best_T, best_C, np.array(norm_term_list), Tc_list


def maxCostMahalanobis_minminmax(M, mu, sqMahalanobis, constraints, X_s, X_t, c_center=None, dim_M=False):
    problemAlgo = cvx.Problem(cvx.Maximize(mu), constraints=constraints)
    # problemAlgo.solve(solver='MOSEK', warm_start=True, verbose=False,
    #                   mosek_params={'MSK_IPAR_INTPNT_MAX_ITERATIONS': 10000, 'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-12,
    #                                 'MSK_DPAR_DATA_TOL_X': 1e-12, 'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-12,
    #                                 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-12})
    # problemAlgo.solve(solver='OSQP', warm_start=True, verbose=False)
    problemAlgo.solve(solver='SCS', warm_start=True, verbose=False, max_iters=100000, eps=1e-8)
    # problemAlgo.solve(solver= 'CVXOPT', warm_start= True, verbose= False, max_iter= 10000000, abstol= 1e-12, feastol= 1e-12)
    M = M.value
    C_star = sqMahalanobisMat(X_s, X_t, M, cvxpy=False) + c_center
    dualVars = np.array([c.dual_value for c in constraints[1:]]).reshape(-1)
    if dim_M:
        return mu.value, C_star, dualVars, M
    else:
        return mu.value, C_star, dualVars


def updateMahalanobis_minminmax(newW, dualVars, weightMats, M, mu, sqMahalanobis, constraints, X_s, X_t, c_center=None):
    toKeep = dualVars > 1e-12
    weightMats = weightMats[toKeep]
    weightMats = vstack((weightMats, csr_matrix(newW.reshape((1, newW.shape[0] * newW.shape[1])))))
    constraints = [constraints[0]] + [c for c, keep in zip(constraints[1:], toKeep) if keep]
    constraints.append(cvx.trace(newW.toarray().T * (sqMahalanobis + c_center)) >= mu)
    return dualVars[toKeep], weightMats, M, mu, sqMahalanobis, constraints, X_s, X_t, c_center


"""
    Solves the max mu problem in the case of a finite set of cost matrices, or equivalently their convex combinations
    arguments:
        * G: <P,C> matrix for the current finite set of transport matrices P
        * costMats: the list of cost matrices
        * weightMats: needed 
"""


def maxCostFinite(G, costMats):
    p = cvx.Variable(G.shape[1], nonneg=True)
    problemAlgo = cvx.Problem(cvx.Minimize(cvx.sum(p)), constraints=[G @ p >= 1])
    problemAlgo.solve(solver='MOSEK', verbose=False)
    if problemAlgo.status == "infeasible" or p.value is None:
        return None, None, None
    p = p.value
    C_star = np.einsum('k, kij', p / np.sum(p), costMats)
    dualVars = problemAlgo.constraints[0].dual_value
    return 1 / problemAlgo.value, C_star, dualVars


def maxCostFiniteReg(G, costMats):
    p = cvx.Variable(G.shape[1], nonneg=True)
    problemAlgo = cvx.Problem(cvx.Minimize(cvx.sum(p)), constraints=[G @ p >= 1])
    problemAlgo.solve(solver='MOSEK', verbose=False)
    if problemAlgo.status == "infeasible" or p.value is None:
        return None, None, None

    p = p.value
    C_star = np.einsum('k, kij', p / np.sum(p), costMats)
    if C_star is None:
        print("WARNING!")
    dualVars = problemAlgo.constraints[0].dual_value.flatten()
    return 1 / problemAlgo.value, C_star, dualVars


"""
    updates an iteration of the minimax algorithm for the case of finite set of cost matrices
    Implements constraint elimination
"""


def updateFinite(newW, dualVars, weightMats, G, costMats):
    toKeep = dualVars > 1e-12  # keep only binding constraints
    weightMats = weightMats[toKeep]
    weightMats = vstack((weightMats, csr_matrix(newW.reshape((1, newW.shape[0] * newW.shape[
        1])))))  # this reshaping is necessary as there's no stacking function on the 3rd axis for sparse format
    G = G[toKeep]
    G = np.vstack((G, np.einsum("ij, lij -> l", newW.toarray(),
                                costMats)))  # local conversion from sparse to dense, in order to use einsum function
    return dualVars[toKeep], weightMats, G, costMats


def updateFiniteReg(newW, dualVars, weightMats, G, costMats):
    toKeep = dualVars > 1e-12  # keep only binding constraints
    weightMats = np.vstack((weightMats, newW.reshape((1, newW.shape[0] * newW.shape[
        1]))))  # this reshaping is necessary as there's no stacking function on the 3rd axis for sparse format
    G = G[toKeep]
    G = np.vstack((G, np.einsum("ij, lij -> l", newW,
                                costMats)))  # local conversion from sparse to dense, in order to use einsum function
    return dualVars[toKeep], weightMats, G, costMats


def maxCostLowRank(M, sqMahalanobis, constraints, X_s, X_t, rank):
    problemAlgo = cvx.Problem(cvx.Minimize(cvx.trace(M)), constraints=constraints)
    problemAlgo.solve(solver='MOSEK')
    M = M.value
    traceM = np.trace(M)
    #    M *= rank/traceM
    C_star = sqMahalanobisMat(X_s, X_t, M * rank / traceM, cvxpy=False)


#
#    return rank/traceM, C_star, [c.dual_value for c in constraints] 
#
def updateLowRank(newW, mu, constVals, M, sqMahalanobis, constraints, X_s, X_t, rank):
    constraints = [constraints[0]] + [c for c in constraints[1:] if c.dual_value > 1e-12]
    constraints.append(cvx.trace(newW.T * sqMahalanobis) >= 1)
    return dualVars, weightMats, M, sqMahalanobis, constraints, X_s, X_t, rank


def maxCostMahalanobisSVM(data, y_data, stBox, d, m, n, dim_M=False):
    #    linsvc = LinearSVC(penalty= 'l2', loss= 'hinge', C= 1e10, max_iter= 1e8, tol= 1e-12, fit_intercept= False, dual= True)
    linsvc = SVC(kernel='linear', C=10, max_iter=-1, tol=1e-10)
    #    linsvc = LogisticRegression(penalty= 'l2',  C= 1000, max_iter= 1e8, tol= 1e-10, fit_intercept= False, dual= False, solver= "lbfgs", warm_start= True)
    try:
        linsvc.fit(data, y_data)
    except ValueError:
        print("!")
    M = linsvc.coef_.flatten()

    normM = np.linalg.norm(M)
    C_star = np.dot(stBox, M / normM).reshape(m, n)
    dualVars = np.zeros(len(data))
    dualVars[linsvc.support_] = linsvc.dual_coef_.flatten()
    if dim_M:
        M_plot = M.reshape(d, d)
        return 1 / normM, C_star, dualVars, M_plot
    else:
        return 1 / normM, C_star, dualVars


def updateMahalanobisSVM(newW, dualVars, weightMats, data, y_data, stBox, d, m, n):
    toKeep = np.abs(dualVars) > 1e-12
    #    toKeep[::2]= np.logical_or(toKeep[::2], toKeep[1::2])
    #    toKeep[1::2]= np.logical_or(toKeep[::2], toKeep[1::2])
    try:
        weightMats = weightMats[toKeep]
    except IndexError:
        print("WARNING")

    data = data[toKeep]
    y_data = y_data[toKeep]

    newW = csr_matrix(newW.reshape((1, newW.shape[0] * newW.shape[1])))
    weightMats = vstack((weightMats, newW, -newW))

    newData = np.dot(newW.toarray(), stBox)
    #    print("before  constraint elimination: %d" %len(data))

    #    print("after  constraint elimination: %d" %len(data))
    data = np.vstack((data, newData, -newData))
    #    print(len(data))
    y_data = np.append(y_data, (1, -1))
    return dualVars[toKeep], weightMats, data, y_data, stBox, d, m, n


# from scipy.spatial.distance import
def maxCostMahalanobisDual(V, X_s, X_t, p=None):
    p = cvx.Variable(V.shape[0], nonneg=True)
    problem = cvx.Problem(objective=cvx.Minimize(cvx.quad_form(p, np.dot(V, V.T))), constraints=[cvx.sum(p) == 1])
    # problem.solve(solver='OSQP', max_iter=100000000, eps_abs=1e-12, eps_rel=1e-14)
    #    problem.solve(solver= 'ECOS', max_iters=100000000, abstol= 1e-10, reltol= 1e-10, feastol= 1e-10)
    problem.solve(solver= 'SCS', max_iters=100000000, eps= 1e-12, use_indirect= False)
    p = p.value

    d = X_s.shape[1]
    M = np.dot(p, V).reshape((d, d))
    normM = np.linalg.norm(M)
    C = sqMahalanobisMat(X_s, X_t, M / normM, cvxpy=False)
    return normM, C, p


# def maxCostMahalanobisDualFW(V, X_s, X_t, p=None):
#     t = V.shape[0]
#     Q = np.dot(V, V.T)
#     if p is None:
#         p = np.ones(1)
#     else:
#         p = np.hstack((p, 0))
#         for k in range(1, 1000):
#             s = np.zeros(t)
#             s[np.argmin(np.dot(Q, p))] = 1
#             delta = s - p
#             Qd = np.dot(Q, delta)
#             gamma = np.clip(-np.dot(p, Qd) / np.dot(delta, Qd), 0, 1)
#             #            if np.isnan(gamma):
#             #                gamma = 0.5
#
#             err = gamma * np.max(np.abs(delta))
#             #        print(err)
#             if err < 1e-15: break
#             p = p + gamma * delta
#     # print(p)
#     # print(sum(p))
#     d = X_s.shape[1]
#     M = np.dot(p, V).reshape((d, d))
#     normM = np.linalg.norm(M)
#     C = sqMahalanobisMat(X_s, X_t, M / normM, cvxpy=False)
#     return normM, C, p


def updateMahalanobisDual(newW, dualVars, weightMats, V, X_s, X_t, p):
    toKeep = dualVars > 0
    # print(min(dualVars))
    weightMats = vstack((weightMats[toKeep], csr_matrix(newW.reshape((1, newW.shape[0] * newW.shape[1])))))
    V = np.vstack((V[toKeep], vecDisplacementMatrix(X_s, X_t, newW)))
    dualVars = dualVars[toKeep]
    dualVars /= np.sum(dualVars)
    #    print(weightMats)
    #    print(len(dualVars[toKeep]))
    return dualVars, weightMats, V, X_s, X_t, dualVars


def vecDisplacementMatrix(X_s, X_t, W):
    d = X_s.shape[1]
    weights, inds = weightsAndInds(W)

    diffs = X_s[inds[0]] - X_t[inds[1]]
    prods = np.einsum('ij,ik->ijk', diffs, diffs).reshape(len(weights), d ** 2)

    return np.dot(weights, prods)

    # resMat = np.zeros((d,d))
    # for w, i, j in zip(*((weights, )+ tuple(inds))):
    #     diff = X_s[i] - X_t[j]
    #     resMat += w*np.outer(diff,diff)
    # return resMat.reshape(d**2)


def updateMahalanobis(newW, dualVars, weightMats, M, sqMahalanobis, constraints, X_s, X_t):
    toKeep = dualVars > 0
    weightMats = weightMats[toKeep]
    weightMats = vstack((weightMats, csr_matrix(newW.reshape((1, newW.shape[0] * newW.shape[1])))))
    constraints = [c for c, keep in zip(constraints, toKeep) if keep]
    constraints.append(cvx.trace(newW.toarray().T * sqMahalanobis) >= 1)
    return dualVars[toKeep], weightMats, M, sqMahalanobis, constraints, X_s, X_t


def maxCostMahalanobis(M, sqMahalanobis, constraints, X_s, X_t):
    problemAlgo = cvx.Problem(cvx.Minimize(cvx.sum(M ** 2)), constraints=constraints)
    problemAlgo.solve(solver='MOSEK', warm_start=True, verbose=False)
    M = M.value
    normM = np.linalg.norm(M)
    C_star = sqMahalanobisMat(X_s, X_t, M / normM, cvxpy=False)
    dualVars = np.array([c.dual_value for c in constraints])
    return 1 / normM, C_star, dualVars


def updateMahalanobis(newW, dualVars, weightMats, M, sqMahalanobis, constraints, X_s, X_t):
    toKeep = dualVars > 1e-12
    weightMats = weightMats[toKeep]
    weightMats = vstack((weightMats, csr_matrix(newW.reshape((1, newW.shape[0] * newW.shape[1])))))
    constraints = [c for c, keep in zip(constraints, toKeep) if keep]
    #    print("after  constraint elimination: %d" %len(constraints))
    constraints.append(cvx.trace(newW.toarray().T * sqMahalanobis) >= 1)
    return dualVars[toKeep], weightMats, M, sqMahalanobis, constraints, X_s, X_t


def sqMahalanobisMat(X_s, X_t, M, cvxpy=True):
    if cvxpy:
        diagM_s = cvx.diag(X_s * M * X_s.T)[:, None]
        diagM_t = cvx.diag(X_t * M * X_t.T)[None, :]
        XsMt = X_s * M * X_t.T
        return diagM_s * np.ones(len(X_t))[None, :] + np.ones(len(X_s))[:, None] * diagM_t - 2 * XsMt
    else:
        diagM_s = np.diag(np.linalg.multi_dot((X_s, M, X_s.T)))
        diagM_t = np.diag(np.linalg.multi_dot((X_t, M, X_t.T)))
        XsMt = np.linalg.multi_dot((X_s, M, X_t.T))
        return diagM_s[:, None] + diagM_t - 2 * XsMt


def differences(X_s, X_t):
    X_s_rep = np.repeat(X_s, len(X_t), axis=0)
    X_t_rep = np.tile(X_t, (len(X_s), 1))
    return X_s_rep - X_t_rep


"""
    Takes a transport matrix and outputs the indices and weights of non null instances. Supports sparse format
"""


def weightsAndInds(W):
    if issparse(W):
        indices = W.indices
        indptr = W.indptr
        inds = np.array([[i, j] for i in range(W.shape[0]) for j in indices[indptr[i]:indptr[i + 1]]]).T
        return W.data.copy(), inds
    else:
        logicInds = W > 0
        inds = np.where(logicInds)
        return W[logicInds], np.array(inds)


from matplotlib.collections import LineCollection
def plotTransport(X_s, X_t, W, title):
    
    weights, inds = weightsAndInds(W)
    # weights = weights **3
    weights /= np.max(weights)
    fig, ax = plt.subplots()
    #    plt.subplot(1,2,1)
    labels = np.hstack((np.ones(len(X_s)), -np.ones(len(X_t))))
    
    ax.scatter(*np.vstack((X_s[:, :2], X_t[:,:2])).T, edgecolors= "k", c= labels, cmap= "Paired")
   
    # with a for loop, slow
    # for ind_s, ind_t, weight in zip(*(tuple(inds) + (weights,))):
    #     plt.plot([X_s[ind_s, 0], X_t[ind_t, 0]], [X_s[ind_s, 1], X_t[ind_t, 1]], color='k', alpha= 0.5*weight)
    # plt.title(title)
    # with Matplotlib's LineCollection
    # segments = [np.column_stack([[X_s[ind_s, 0], X_t[ind_t, 0]], [X_s[ind_s, 1], X_t[ind_t, 1]]]) for ind_s, ind_t in zip(*tuple(inds))]
    segments = np.moveaxis(np.dstack((X_s[inds[0],:2], X_t[inds[1],:2])),1,2)
    line_segments = LineCollection(segments, colors= np.hstack((np.zeros((len(weights),3)), 0.3*weights[:,None]**5)), rasterized= True)
    ax.add_collection(line_segments)
    
    ax.set_aspect("equal")
    ax.set_xticks(())
    ax.set_yticks(())
    fig.tight_layout()
    return None

"""
    Hölder conjugate
"""
def dualOrder(p):
    if p == 1: return np.inf
    if p == np.inf: return 1
    return p/(p-1)


"""
    Computes a vector v that satisfies <u,v> = norm(u, order)
    i.e the vector of the equality case of the Hölder inequality
"""
def HolderArgmax(u, order=2, truncate=None):
    dOrder = dualOrder(order)

    # if order == 2:
    #     return unit_u
    # WARNING, here u is assumed to be sorted in the ascending order
    if truncate is not None:
        u[:-truncate] = 0

    # Treat order = 1 in particular to avoid 0**0 = 1
    if order == 1 and truncate is not None:
        unit_u = np.zeros_like(u)
        unit_u[-truncate:] = 1
        return unit_u
    unit_u = u / np.linalg.norm(u, ord=order)
    return unit_u ** (order - 1)  # order/dOrder = order-1


"""
    Computes a matrix N that satisfies <M,N> = schatten_norm(M, order= p), where M est a PSD matrix
    i.e the matrix of the equality case of the Hölder inequality version for matrices and
    Schatten p-norms and truncated Schatten p-norm
"""
def HolderArgmaxSchatten(M, order=2, truncate=None):
    if order == 2:
        return M / np.linalg.norm(M)
    eigvals, P = np.linalg.eigh(M)
    eigvals = np.abs(eigvals)  # to enforce positivity
    argmaxEigvals = HolderArgmax(eigvals, order=order, truncate=truncate)

    # eventually truncate at order truncate
    # if truncate is not None:
    #     argmaxEigvals[:-truncate] = 0
    return np.linalg.multi_dot((P, np.diag(argmaxEigvals), P.T))


def maxCostMahalanobisDualFW(V, X_s, X_t, order=2, probas=None, dim_M=False):
    d = X_s.shape[1]
    if probas is None:
        probas = np.ones(1)
    else:
        # probas = np.ones(len(probas)+1)
        probas = np.hstack((probas, 1))
        probas /= np.sum(probas)
    gap = 1e10
    maxLow = 0
    i = 0
    while gap >= 1e-10 and i <= 1000:
        V_comb = np.dot(probas, V).reshape((d, d))
        argmaxVmat = HolderArgmaxSchatten(V_comb, order=order)
        grad = np.dot(V, argmaxVmat.flatten())
        eigV_comb = np.linalg.eigvalsh(V_comb)
        normV_comb = np.linalg.norm(eigV_comb, ord=order)

        # new wieghts of few transport plans
        probas_aux = np.zeros(len(grad))
        probas_aux[np.argmin(grad)] = 1
        diff = probas_aux - probas
        tau = 2 / (i + 2)
        probas += tau * diff

        low = np.dot(probas, grad)
        maxLow = max(maxLow, low)
        gap = np.abs(normV_comb - maxLow)  # /maxLow

        i += 1
    # print(i)
    # print("aux_gap= %e" % gap)
    d = X_s.shape[1]
    C = sqMahalanobisMat(X_s, X_t, argmaxVmat, cvxpy=False)
    if dim_M:
        return normV_comb, C, probas, argmaxVmat
    else:
        return normV_comb, C, probas

def updateMahalanobisDualFW(newW, dualVars, weightMats, V, X_s, X_t, order, probas):
    toKeep = dualVars > 0
    weightMats = vstack((weightMats[toKeep], csr_matrix(newW.reshape((1, newW.shape[0] * newW.shape[1])))))
    V = np.vstack((V[toKeep], vecDisplacementMatrix(X_s, X_t, newW)))
    dualVars = dualVars[toKeep]
    dualVars /= np.sum(dualVars)
    return dualVars, weightMats, V, X_s, X_t, order, dualVars