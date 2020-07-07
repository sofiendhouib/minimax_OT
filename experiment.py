#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sys import path
import imageio
import sklearn.metrics.pairwise
import sklearn.datasets
from skimage import img_as_ubyte
import scipy.stats
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cdist

import ot_max_funcs as otMax

from ot.lp import emd
from ot.bregman import sinkhorn_epsilon_scaling
from ot import da, sinkhorn

path.append("./SubspaceRobustWasserstein")  # if local import
from SRW import SubspaceRobustWasserstein  # Change the path above
from Optimization.frankwolfe import FrankWolfe


# ------------------------- hypercube ----------------------------------


def hypercube_M(d=30, n=250, k_star=2, k_star_max=21, treshold=0.12, nb_iter=100, maxIter=10,
                path="pickle/hypercube.pickle"):
    pickle_in = open(path, "wb")

    if k_star_max is None:
        k_star_max = k_star + 1
    from scipy.sparse import csr_matrix, vstack

    def T(x, d, dim=2):
        assert dim <= d
        assert dim >= 1
        assert dim == int(dim)
        return x + 2 * np.sign(x) * np.array(dim * [1] + (d - dim) * [0])

    def fragmented_hypercube(n, d, dim):
        assert dim <= d
        assert dim >= 1
        assert dim == int(dim)

        a = (1. / n) * np.ones(n)
        b = (1. / n) * np.ones(n)

        # First measure : uniform on the hypercube
        X = np.random.uniform(-1, 1, size=(n, d))

        # Second measure : fragmentation
        Y = T(np.random.uniform(-1, 1, size=(n, d)), d, dim)

        return a, b, X, Y

    M_plot = np.zeros((nb_iter, k_star_max - k_star, d, d))
    M_rank = np.zeros((nb_iter, k_star_max - k_star))
    eig = np.zeros((nb_iter, k_star_max - k_star, d))

    pickle.dump(k_star, pickle_in)
    pickle.dump(k_star_max, pickle_in)
    pickle.dump(d, pickle_in)

    for k in range(k_star, k_star_max):
        print("\n k :", k)
        for i in range(nb_iter):
            # Same seed for each k
            np.random.seed(1245 * i + 6789)
            a, b, cube, transformCube = fragmented_hypercube(n, d, k)

            ones = np.ones(n)

            weightMats = csr_matrix(np.outer(ones / n, ones / n).reshape(1, n * n))

            stBox = np.einsum('ij, ik -> ijk', otMax.differences(cube, transformCube),
                              otMax.differences(cube, transformCube)).reshape((n * n, d * d))
            data = np.dot(weightMats.toarray(), stBox)
            data = np.vstack((data, -data))
            y_data = np.array([1, -1])

            weightMats = vstack((weightMats, -weightMats))
            W, error, muValue, C_star, M_plot[i, k - k_star] = otMax.minimaxOT(ones / n, ones / n, maxIter=maxIter,
                                                                               threshold=1e-8,
                                                                               weightMats=weightMats,
                                                                               maxCost=otMax.maxCostMahalanobisSVM,
                                                                               updateFunc=otMax.updateMahalanobisSVM,
                                                                               arguments=(data, y_data, stBox, d, n, n),
                                                                               verbose=False,
                                                                               output=3)
            # Store all the Mahalanobis matrix
            M_plot[i, k - k_star] = M_plot[i, k - k_star] / ((M_plot[i, k - k_star] ** 2).sum()) ** 0.5

            # Store the rank of M
            M_rank[i, k - k_star] = np.linalg.matrix_rank(M_plot[i, k - k_star], tol=treshold)

            #
            M_eig = np.linalg.eigh(M_plot[i, k - k_star])[0]
            M_eig[::-1].sort()
            eig[i, k - k_star] = M_eig

        y = np.mean(eig[:, k - k_star], axis=0)
        std = np.std(eig[:, k - k_star], axis=0)

        pickle.dump(y, pickle_in)
        pickle.dump(std, pickle_in)
    pickle.dump(np.mean(M_rank, axis=0), pickle_in)
    pickle.dump(M_plot, pickle_in)
    pickle_in.close()


# --------------------- Noise sensitivity -------------------------


def creat_matrix_cost(Xs, Xt, type_cost=None, range_euclidian=None, reg=None, eta=None, a=None, b=None,
                      labels=None, number_L=None):
    C_center = np.array([])
    if "euclidian_projection" in type_cost:
        for e in range_euclidian:
            for i in range(number_L):
                np.random.seed(i * 12345 + 6789 + e)
                a, b = Xs.shape[1], np.random.randint(Xs.shape[1] - 1) + 2
                L = (np.random.rand(a, b) - 0.5) * 4
                c = ((np.abs((Xs @ L)[:, np.newaxis, :] - (Xt @ L)[np.newaxis, :, :])) ** e).sum(axis=2) ** (1 / e)
                C_center = np.concatenate((C_center, c[np.newaxis]), axis=0) if C_center.size else c[np.newaxis]
    if "rdm" in type_cost:
        for i in range(number_L):
            np.random.seed(i * 123 + 456789)
            c = np.random.rand(len(Xs), len(Xt))
            C_center = np.concatenate((C_center, c[np.newaxis]), axis=0) if C_center.size else c[np.newaxis]
    if "euclidian" in type_cost:
        for i in range_euclidian:
            c = ((np.abs(Xs[:, np.newaxis, :] - Xt[np.newaxis, :, :]) ** i).sum(axis=2)) ** (1 / i)
            C_center = np.concatenate((C_center, c[np.newaxis]), axis=0) if C_center.size else c[np.newaxis]
    if "projection" in type_cost:
        for i in range(Xs.shape[1]):
            for j in range(i + 1, Xs.shape[1]):
                if i != j:
                    c = (((Xs[:, np.newaxis, [i, j]] - Xt[np.newaxis, :, [i, j]]) ** 2).sum(axis=2)) ** (1 / 2)
                    C_center = np.concatenate((C_center, c[np.newaxis]), axis=0) if C_center.size else c[np.newaxis]
    if "entropy" in type_cost:
        for i in range(len(reg)):
            c = ((((Xs[:, np.newaxis, :] - Xt[np.newaxis, :, :]) ** 2).sum(axis=2)) ** (1 / 2))
            C_center = np.concatenate((C_center, c[np.newaxis]), axis=0) if C_center.size else c[np.newaxis]
    if "PCA" in type_cost:
        for i in range(len(Xs)):
            c = ((((Xs[i][:, np.newaxis, :] - Xt[i][np.newaxis, :, :]) ** 2).sum(axis=2)))
            C_center = np.concatenate((C_center, c[np.newaxis]), axis=0) if C_center.size else c[np.newaxis]

    return C_center


def gaussian_mnist(n=100, m=100, d=10, dataset="gaussian",
                   type_of_cost_matrix=["euclidian_projection"],
                   range_euclidian=[2, 3, 4, 5, 10],
                   reg=[0],
                   ball_radius=0.01,
                   max_iter=10,
                   nb_iter=200,
                   number_L=10,
                   distance_between_gaussian=3,
                   reduction="UMAP",
                   n1=0,
                   n2=1,
                   normalize="C_norm",
                   modif="mahalanobis",
                   threshold=1e-20):
    np.random.seed(12345789)
    if dataset == "gaussian":
        m1 = np.zeros(d)
        cov1 = np.eye(d)
        m2 = np.ones(d) * distance_between_gaussian
        cov2 = np.eye(d)
        Xs = np.random.multivariate_normal(mean=m1, cov=cov1, size=n)
        Xt = np.random.multivariate_normal(mean=m2, cov=cov2, size=m)
    elif dataset == "mnist":
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        y = np.array(y).astype(np.float)
        if reduction == "UMAP":
            y_2 = y[(y == n1) + (y == n2)]
            import umap
            if not os.path.exists("pickle/umap" + str(d) + "_" + str(n1) + "_" + str(n2) + ".pickle"):
                reducer = umap.UMAP(n_components=d, random_state=42)
                embedding = reducer.fit_transform(X[(y == n1) + (y == n2)])
                pickle_in = open("pickle/umap" + str(d) + "_" + str(n1) + "_" + str(n2) + ".pickle", "wb")
                pickle.dump(embedding, pickle_in)
                pickle_in.close()
            else:
                pickle_in = open("pickle/umap" + str(d) + "_" + str(n1) + "_" + str(n2) + ".pickle", "rb")
                embedding = pickle.load(pickle_in)
                pickle_in.close()
            Xs = embedding[y_2 == n1][:n]
            Xt = embedding[y_2 == n2][:m]
        elif reduction == "PCA":
            from sklearn.decomposition import PCA
            PCA_X = PCA(n_components=d)
            PCA_X.fit(X[y < 2])
            components = PCA_X.components_
            Xs = X[y == 0][:n] @ components[:d].T
            Xt = X[y == 1][:m] @ components[:d].T

    plt.plot(Xs[:, 0], Xs[:, 1], "r+")
    plt.plot(Xt[:, 0], Xt[:, 1], "b+")
    plt.show()

    C_center = creat_matrix_cost(Xs, Xt, type_cost=type_of_cost_matrix, range_euclidian=range_euclidian,
                                 reg=reg,
                                 number_L=number_L)
    c_euclidean = (((Xs[:, np.newaxis, :] - Xt[np.newaxis, :, :])) ** 2).sum(axis=2) ** 0.5

    C_center = np.vstack((C_center, c_euclidean[np.newaxis, :]))
    T = []
    W_init = np.zeros(len(C_center))
    norm_term = np.zeros(len(C_center))
    for i in range(len(C_center)):
        if reg[0] is 0:
            T.append(emd(np.ones(n) / n, np.ones(m) / m, C_center[i]))
        else:
            T.append(sinkhorn_epsilon_scaling(np.ones(n) / n, np.ones(m) / m, C_center[i], reg=reg[0]))
        if normalize == "OT":
            norm_term[i] = np.sum(T[i] * C_center[i])
        elif normalize == "C_norm":
            norm_term[i] = np.linalg.norm(C_center[i])
        elif normalize == "max":
            norm_term[i] = np.max(C_center[i])
        # C_center is normalize
        C_center[i] = C_center[i] / norm_term[i]
        # Store the Wasserstein distance of the origine
        W_init[i] = (T[i] * C_center[i]).sum()

    worst_C_center, worst_T, worst_C, norm_term_list, Tc_list = otMax.miniminimaxOT(r=np.ones(n) / n,
                                                                                    c=np.ones(m) / m,
                                                                                    maxIter=max_iter,
                                                                                    threshold=threshold,
                                                                                    maxCost=otMax.maxCostMahalanobis_minminmax,
                                                                                    updateFunc=otMax.updateMahalanobis_minminmax,
                                                                                    Xt=Xt,
                                                                                    Xs=Xs,
                                                                                    C_center=C_center,
                                                                                    verbose=False,
                                                                                    reg=reg,
                                                                                    uniform_init=True,
                                                                                    ball_radius=ball_radius,
                                                                                    normalize=normalize)  # = r**2
    print("")
    worst_W = (worst_C * worst_T).sum(axis=(1, 2))

    # Wasserstein distance after modification of C_center
    W = np.zeros((nb_iter, len(C_center)))

    # Keep only the stable value
    stable = np.array([True] * len(C_center))

    # Loop over the number of iteration of modification of the cost matrix
    for iter_j in range(nb_iter):
        print(".", end="")
        if modif == "mahalanobis":
            # Rdm matrix M with norm egal to ball_radius
            np.random.seed(iter_j * 1234 + 56789)
            a, b = Xs.shape[1], np.random.randint(Xs.shape[1] - 1) + 2
            L = (np.random.rand(a, b) - 0.5) * 4
            M = L @ L.T
            M = ball_radius * M / np.sum(M ** 2)
            c = otMax.sqMahalanobisMat(Xs, Xt, M, cvxpy=False) ** 0.5
            # New cost matrix with a small modification
            C_center_j = C_center + c[np.newaxis, :, :]
        elif modif == "rdm":
            print("WARNING : this part of the code may not give the expected result, only mahalanobis as been tested")
            c = np.random.rand(len(Xs), len(Xt)) * ball_radius
            # New cost matrix with a small modification
            C_center_j = C_center + c[np.newaxis, :, :]
        elif modif == "sample":
            print("WARNING : this part of the code may not give the expected result, only mahalanobis as been tested")
            a = 1  # How much sample we take
            if dataset == "gaussian":
                Xs = np.random.multivariate_normal(mean=m1, cov=cov1, size=a * n)
                Xt = np.random.multivariate_normal(mean=m2, cov=cov2, size=a * m)
            elif dataset == "mnist":
                if reduction == "UMAP":
                    y_2 = y[(y == n1) + (y == n2)]
                    Xs = embedding[y_2 == n1][(1 + iter_j) * n:(1 + iter_j + 1) * n]
                    Xt = embedding[y_2 == n2][(1 + iter_j) * m:(1 + iter_j + 1) * m]
                elif reduction == "PCA":
                    Xs = X[y == n1][(1 + iter_j) * n:(1 + iter_j + 1) * n] @ components[:d].T
                    Xt = X[y == n2][(1 + iter_j) * m:(1 + iter_j + 1) * m] @ components[:d].T
            C_center_j = creat_matrix_cost(Xs, Xt, type_cost=type_of_cost_matrix, range_euclidian=range_euclidian,
                                           reg=reg,
                                           number_L=number_L)
            C_center_j = C_center_j / norm_term[:, np.newaxis, np.newaxis]

        for i in range(len(C_center_j)):
            if reg[0] == 0:
                T_i = emd(np.ones(n) / n, np.ones(m) / m, C_center_j[i])
            else:
                T_i = sinkhorn_epsilon_scaling(np.ones(n) / n, np.ones(m) / m, C_center_j[i], reg=reg[0])
            if np.sum(T_i) < 1 - 0.001:
                stable[i] = False
                print("Not stable", (T_i * C_center_j[i]).sum(), i)
            # Wasserstein distance after modification of C_center
            W[iter_j, i] = (T_i * C_center_j[i]).sum()

    worst_W_ = worst_W - W_init
    W_ = (np.abs(W - W_init)).mean(axis=0)
    order_plot = np.argsort(worst_W_)
    worst_W_stable = worst_W_[order_plot][stable[order_plot]]  # wasserstein stability
    W_stable = W_[order_plot][stable[order_plot]]  # Noise stability

    pickle_in = open("pickle/" + dataset + ".pickle", "wb")
    pickle.dump(W_stable, pickle_in)
    pickle.dump(worst_W_, pickle_in)
    pickle.dump(worst_W_stable, pickle_in)
    pickle.dump(stable, pickle_in)
    pickle.dump(order_plot, pickle_in)
    pickle.dump(W_, pickle_in)
    pickle_in.close()

    return None


# --------------------COLOR TRANSFER-----------------------

def color_matrix_cost(range_euclidian, number_L, Xs, Xt, ball_radius=None):
    if ball_radius is None:
        C_center = (((np.abs(Xs[:, np.newaxis, :] - Xt[np.newaxis, :, :])) ** 2).sum(axis=2) ** (1 / 2))[np.newaxis]
        cs = np.random.rand(number_L * len(range_euclidian), len(Xs), len(Xt))
        C_center = np.concatenate((C_center, cs))
        for e in range_euclidian:
            for i in range(number_L):
                np.random.seed(i * 1234 + 56789 + e)
                #             L = (np.random.rand(Xs.shape[1], np.random.randint(1, 4)) - 0.5) * 4
                L = (np.random.rand(Xs.shape[1], Xs.shape[1]) - 0.5) * 4
                c = ((np.abs((Xs @ L)[:, np.newaxis, :] - (Xt @ L)[np.newaxis, :, :])) ** e).sum(axis=2) ** (1 / e)
                C_center = np.concatenate((C_center, c[np.newaxis]), axis=0) if C_center.size else c[np.newaxis]
        return C_center
    else:
        np.random.seed(12345)
        L = (np.random.rand(Xs.shape[1], Xs.shape[1]) - 0.5)
        L = 1 * ball_radius * L / np.sum(L ** 2)
        C_center = sklearn.metrics.pairwise.euclidean_distances(Xs @ L, Xt @ L)[np.newaxis]
        #         C_center = np.random.rand(Xs.shape[0], Xt.shape[0])
        return C_center


def im2mat(I):
    """Converts an image to matrix (one pixel per line)"""
    return I.reshape((I.shape[0] * I.shape[1], I.shape[2]))


def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)


def minmax(I):
    return np.clip(I, 0, 1)


def load_images(path):
    I1 = imageio.imread(path[0]).astype(np.float64) / 256
    I2 = imageio.imread(path[1]).astype(np.float64) / 256
    return I1, I2


def training_samples(I1, I2, nb=200, sigma=0.1):
    np.random.seed(12345)
    X1 = im2mat(I1)
    X2 = im2mat(I2)

    idx1 = np.random.choice(X1.shape[0], size=(nb,))
    idx2 = np.random.choice(X2.shape[0], size=(nb,))

    Xs = X1[idx1, :]
    Xt = X2[idx2, :]
    return Xs, Xt, X1, X2


def plot_start(Xs, Xt, I1, I2):
    plt.figure(1, figsize=(6.4, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.axis('off')
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.axis('off')
    plt.title('Image 2')
    plt.show()

    plt.figure(2, figsize=(6.4, 3))

    plt.subplot(1, 2, 1)
    plt.scatter(Xs[:, 0], Xs[:, 2], c=Xs)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Red')
    plt.ylabel('Blue')
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.scatter(Xt[:, 0], Xt[:, 2], c=Xt)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Red')
    plt.ylabel('Blue')
    plt.title('Image 2')
    plt.tight_layout()
    plt.show()


def color_transfer(range_euclidian=[2, 4, 6, 8, 10], number_L=10, reg=[0], normalize="OT", ball_radius=0.001,
                   threshold=1e-8, max_iter=200, nb=200, plot=False,
                   path=['color_transfer/Images/ocean_day.jpg', 'color_transfer/Images/ocean_sunset.jpg'],
                   path_pickle="pickle/sunset.pickle",
                   path_save_images='color_transfer/Images_saved/sunset'):
    np.random.seed(123456789)

    I1, I2 = load_images(path=path)
    Xs, Xt, X1, X2 = training_samples(I1, I2, nb=nb)
    plot_start(Xs, Xt, I1, I2)

    n, m = len(Xs), len(Xt)
    C_center = color_matrix_cost(range_euclidian, number_L, Xs, Xt)

    worst_C_center, worst_T, worst_C, norm_term_list, Tc_list = otMax.miniminimaxOT(r=np.ones(n) / n,
                                                                                    c=np.ones(m) / m,
                                                                                    maxIter=max_iter,
                                                                                    threshold=threshold,
                                                                                    maxCost=otMax.maxCostMahalanobis_minminmax,
                                                                                    updateFunc=otMax.updateMahalanobis_minminmax,
                                                                                    Xt=Xt,
                                                                                    Xs=Xs,
                                                                                    C_center=C_center,
                                                                                    verbose=False,
                                                                                    reg=reg,
                                                                                    uniform_init=True,
                                                                                    ball_radius=ball_radius,
                                                                                    normalize=normalize)

    worst_W = (worst_C * worst_T).sum(axis=(1, 2))
    diff_W = np.zeros(len(worst_W))

    for i in range(len(worst_W)):
        ot_emd = da.EMDTransport()
        ot_emd.fit(Xs=Xs, Xt=Xt, M=C_center[i])
        if normalize == "OT":
            diff_W[i] = worst_W[i] - 1
            C_center[i] = C_center[i] / np.sum(ot_emd.coupling_ * C_center[i])
        elif normalize == "C_norm":
            diff_W[i] = worst_W[i] - (np.sum(ot_emd.coupling_ * C_center[i]) / np.linalg.norm(C_center[i]))
            C_center[i] = C_center[i] / np.linalg.norm(C_center[i])
        elif normalize == "max":
            diff_W[i] = worst_W[i] - (np.sum(ot_emd.coupling_ * C_center[i]) / np.max(C_center[i]))
            C_center[i] = C_center[i] / np.max(C_center[i])

    loop_order = np.argsort(diff_W)

    print("")
    if plot:
        a = 0
        for i in loop_order.tolist():
            print(".", end="")
            I1_list = []
            # I2_list = []

            # EMDTransport
            ot_emd = da.EMDTransport()
            ot_emd.fit(Xs=Xs, Xt=Xt, M=C_center[i])

            transp_Xs_emd = ot_emd.transform(Xs=X1)
            # transp_Xt_emd = ot_emd.inverse_transform(Xt=X2)

            I1_list.append(minmax(mat2im(transp_Xs_emd, I1.shape)))
            # I2_list.append(minmax(mat2im(transp_Xt_emd, I2.shape)))

            imageio.imwrite(path_save_images + str(a) + "_" + str(i) + '.png', img_as_ubyte(I1_list[0]))
            # imageio.imwrite(path_save_images + "_reverse_" + str(a) + "_" + str(i) + '.png', img_as_ubyte(I2_list[0]))
            a += 1
    pickle_in = open(path_pickle, "wb")
    pickle.dump(diff_W, pickle_in)
    pickle.dump(loop_order, pickle_in)
    pickle_in.close()
    return diff_W, loop_order


# --------------------Impact of q-----------------------


def minmaxFW(x_0, x_1, reg=0.0, order=2.0, verbose=False, thd=1e-7):
    n0, d0 = x_0.shape
    n1, d1 = x_1.shape

    # Uniform distributions on samples
    a = np.ones((n0,)) / n0
    b = np.ones((n1,)) / n1

    if order == np.inf:
        FW = FrankWolfe(reg=reg, step_size_0=None, max_iter=1000, threshold=0.0001,
                        max_iter_sinkhorn=200, threshold_sinkhorn=1e-6, use_gpu=False)
        SRW = SubspaceRobustWasserstein(x_0, x_1, a, b, FW, 1)
        SRW.run()
        gamma = SRW.get_pi()
        return gamma, otMax.sqMahalanobisMat(x_0, x_1, SRW.get_Omega(), cvxpy=False), SRW.get_Omega()
    if reg == 0:
        W0 = emd(a, b, cdist(x_0, x_1)**2)
    else:
        W0 = sinkhorn(a, b, cdist(x_0, x_1)**2, reg=reg)
    weightMats = csr_matrix(W0.reshape(1, n0 * n1))
    if thd > 1000:  # OT
        return W0, cdist(x_0, x_1)**2, np.eye(d0)
    V0 = otMax.vecDisplacementMatrix(x_0, x_1, W0)
    V0 = V0.reshape(1, len(V0))
    W, _, _, C_star, M_star = otMax.minimaxOT(a, b,
                                              maxIter=400,
                                              threshold=thd,
                                              weightMats=weightMats,
                                              maxCost=otMax.maxCostMahalanobisDualFW,
                                              updateFunc=otMax.updateMahalanobisDualFW,
                                              arguments=(V0, x_0, x_1, order, None),
                                              verbose=True, output=3,
                                              reg=reg)
    # gamma, C_star, M = otMax.maxminFW_Mahalanobis_q(a, b, x_0, x_1, M0, sinkhorn_reg=reg, order=order,
    #                                               truncate=None, thd=thd, verbose=verbose, low_memory=False)
    return W.toarray(), C_star, M_star


def generate_dataset(synthetic_case):
    X, y = sklearn.datasets.make_classification(n_samples=20, n_features=20, n_informative=10, n_redundant=0,
                                                n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                                                flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                                                shuffle=True, random_state=None)
    x_0 = X[y == 0]
    x_1 = X[y == 1]

    return x_0, x_1


def main_q(names=[1],
           reg=0.25,
           uniform=False,
           orders=[1, 1.25, 1.5, 1.75, 2, 5, 10, 20, 50, 100]):
    np.random.seed(42)
    for name_index, name in enumerate(names):
        x_0, x_1 = generate_dataset(name)
        gamma_, C_, M_ = minmaxFW(x_0=x_0, x_1=x_1, reg=reg, order=1, verbose=False, thd=np.inf)
        dict_pickle = {"gamma": gamma_, "C": C_, "M": M_}
        with open('pickle/order/' + str(name_index) + "OT" + '.pickle', 'wb') as f:
            pickle.dump(dict_pickle, f)
        print('Graph: ' + str(name + 1))
        for order in orders:
            print("")
            print('Order: ' + str(order))

            gamma_, C_, M_ = minmaxFW(x_0=x_0, x_1=x_1, reg=reg, order=order, verbose=False)
            dict_pickle = {"gamma": gamma_, "C": C_, "M": M_}
            with open('pickle/order/' + str(name_index) + str(order) + '.pickle', 'wb') as f:
                pickle.dump(dict_pickle, f)

        gamma_, C_, M_ = minmaxFW(x_0=x_0, x_1=x_1, reg=reg, order=np.inf, verbose=False)
        dict_pickle = {"gamma": gamma_, "C": C_, "M": M_}
        with open('pickle/order/' + str(name_index) + "inf" + '.pickle', 'wb') as f:
            pickle.dump(dict_pickle, f)


def plot_q(names=[1], orders=[1, 1.25, 1.5, 1.75, 2, 5, 10, 20, 50, 100], treshold_rank=0.09, too_close=10,
           plot_matrix=False):
    n = len(orders)
    matrix_name = ["T", "M", "C"]
    orders_name = ["OT"] + [str(orders[i]) for i in range(len(orders))] + ["SRW"]
    orders_name2 = ["OT q=1"] + ["RKP q=" + str(orders[i]) for i in range(len(orders))] + ["SRW q=" + r"$\infty$"]

    for name_index, name in enumerate(names):
        with open('pickle/order/' + str(name_index) + "OT" + '.pickle', 'rb') as f:
            dict_pickle = pickle.load(f)
        C = dict_pickle["C"]
        if plot_matrix:
            f1, ax1 = plt.subplots(n + 2, 3, figsize=(5 * 3, 5 * n))

            for i in range(len(ax1)):
                for j in range(len(ax1[i])):
                    ax1[i, j].xaxis.set_visible(False)
                    ax1[i, j].yaxis.set_visible(False)
            ax1[0, 0].imshow(dict_pickle["gamma"], cmap=plt.get_cmap('Blues'))
            ax1[0, 1].imshow(dict_pickle["M"], cmap=plt.get_cmap('Blues'))
            ax1[0, 2].imshow(dict_pickle["C"], cmap=plt.get_cmap('Blues'))
            # ax1[0, 0].set_title("sparse :" + str(scipy.stats.entropy(np.ravel(dict_pickle["gamma"]))))
        list_W_distance = [(dict_pickle["gamma"] * C).sum()]
        list_W_distance_M = [(dict_pickle["gamma"] * dict_pickle["C"]).sum()]
        list_entropy = [scipy.stats.entropy(np.ravel(dict_pickle["gamma"]))]
        list_rank = [np.sum(np.linalg.svd(dict_pickle["M"])[1] > treshold_rank)]
        list_eigenvalues = [np.linalg.svd(dict_pickle["M"])[1]]
        for k, order in enumerate(orders):
            with open('pickle/order/' + str(name_index) + str(order) + '.pickle', 'rb') as f:
                dict_pickle = pickle.load(f)
            if plot_matrix:
                ax1[k + 1, 0].imshow(dict_pickle["gamma"], cmap=plt.get_cmap('Blues'))
                ax1[k + 1, 1].imshow(dict_pickle["M"], cmap=plt.get_cmap('Blues'))
                ax1[k + 1, 2].imshow(dict_pickle["C"], cmap=plt.get_cmap('Blues'))
                # ax1[k + 1, 0].set_title("sparse :" + str(scipy.stats.entropy(np.ravel(dict_pickle["gamma"]))))
            list_W_distance_M.append((dict_pickle["gamma"] * dict_pickle["C"]).sum())
            list_W_distance.append((dict_pickle["gamma"] * C).sum())
            list_entropy.append(scipy.stats.entropy(np.ravel(dict_pickle["gamma"])))
            list_rank.append(np.sum(np.linalg.svd(dict_pickle["M"])[1] > treshold_rank))
            list_eigenvalues.append(np.linalg.svd(dict_pickle["M"])[1])
        with open('pickle/order/' + str(name_index) + "inf" + '.pickle', 'rb') as f:
            dict_pickle = pickle.load(f)
        list_W_distance.append((dict_pickle["gamma"] * C).sum())
        list_W_distance_M.append((dict_pickle["gamma"] * dict_pickle["C"]).sum())
        list_entropy.append(scipy.stats.entropy(np.ravel(dict_pickle["gamma"])))
        list_rank.append(np.sum(np.linalg.svd(dict_pickle["M"])[1] > treshold_rank))
        list_eigenvalues.append(np.linalg.svd(dict_pickle["M"])[1])
        if plot_matrix:
            ax1[-1, 0].imshow(dict_pickle["gamma"], cmap=plt.get_cmap('Blues'))
            ax1[-1, 1].imshow(dict_pickle["M"], cmap=plt.get_cmap('Blues'))
            ax1[-1, 2].imshow(dict_pickle["C"], cmap=plt.get_cmap('Blues'))
            plt.tight_layout()
            for k in range(len(ax1)):
                for i in range(3):

                    leg = ax1[k, i].plot(0, 0, "-", c="white", label="test")
                    legend_artist = ax1[k, i].legend(leg, [orders_name2[k]], loc=3, facecolor=(1, 1, 0.9),
                                                     framealpha=0.9,
                                                     edgecolor="black",
                                                     handlelength=0, fontsize=20, handletextpad=0)
                    leg = ax1[k, i].plot(0, 0, "-", c="white", label="test")
                    if i == 0:
                        legend_name = ["<P*, C*>=" + str(int(round(list_W_distance_M[k], 0)))]
                        ax1[k, i].legend(leg, legend_name, facecolor=(1, 1, 0.9), framealpha=0.9, edgecolor="black",
                                         handlelength=0, fontsize=20, handletextpad=0)
                    elif i == 1:
                        legend_name = ["Rank of M*=" + str(list_rank[k])]
                        ax1[k, i].legend(leg, legend_name, facecolor=(1, 1, 0.9), framealpha=0.9, edgecolor="black",
                                         handlelength=0, fontsize=20, handletextpad=0)
                    if True:
                        ax1[k, i].add_artist(legend_artist)

                    extent = ax1[k, i].get_window_extent().transformed(f1.dpi_scale_trans.inverted())
                    # Don t know why there is such plot problem... probably because it is the first image
                    if matrix_name[i] + '_' + orders_name[k] == "T_OT":
                        diff = extent.get_points()[0, 0] - 0.8095741661296746
                        extent.get_points()[0, 0] -= diff
                        extent.get_points()[1, 0] += diff
                    f1.savefig('experiment_q_image/' + matrix_name[i] + '_' + orders_name[k] + '.pdf',
                               bbox_inches=extent)  # .expanded(1.1, 1.2))

            f1.savefig("experiment_q_image/Matrix.pdf")
            plt.show()

        # manual log scale
        pos_tick = list(np.log(orders))
        # plt.figure(figsize=(10, 8))
        plt.plot(pos_tick, list_entropy[1:-1], color="black", label="Our")
        plt.axhline(y=list_entropy[0], color="blue", label="OT")
        plt.axhline(y=list_entropy[-1], color="red", label="SRW k=1")
        plt.xlabel("q")
        plt.ylabel("Entropy")
        plt.legend(loc=7)
        plt.xticks(pos_tick, [1] + [""] * (too_close - 1) + orders[too_close:], rotation=0, size=10)
        plt.savefig("experiment_q_image/Entropy.pdf")
        plt.show()

        # plt.figure(figsize=(10, 8))
        plt.plot(pos_tick, list_W_distance[1:-1], color="black", label="Our")
        plt.axhline(y=list_W_distance[0], color="blue", label="OT")
        plt.axhline(y=list_W_distance[-1], color="red", label="SRW k=1")
        plt.xlabel("q")
        plt.ylabel("Wasserstein distance euclidean")
        plt.legend(loc=7)
        plt.xticks(pos_tick, [1] + [""] * (too_close - 1) + orders[too_close:], rotation=0, size=10)
        plt.savefig("experiment_q_image/Wasserstein_distance_with_euclidean.pdf")
        plt.show()

        limit = 8
        # plt.figure(figsize=(10, 8))
        plt.plot(pos_tick[:limit], (list_W_distance_M[1:-1])[:limit], color="black", label="Our")
        plt.axhline(y=list_W_distance_M[0], color="blue", label="OT")
        plt.axhline(y=list_W_distance_M[-1], color="red", label="SRW k=1")
        plt.xlabel("q")
        plt.ylabel("Wasserstein distance")
        plt.legend(loc=7)
        plt.xticks(pos_tick[:limit], ([1] + [""] * (too_close - 1) + orders[too_close:])[:limit], rotation=0, size=10)
        plt.savefig("experiment_q_image/Wasserstein_distance_with_M.pdf")
        plt.show()

        # plt.figure(figsize=(10, 8))
        plt.plot(pos_tick, list_rank[1:-1], color="black", label="Our")
        plt.axhline(y=list_rank[0], color="blue", label="OT")
        plt.axhline(y=1, color="red", label="SRW k=1")
        plt.xlabel("q")
        plt.ylabel("Rank of M")
        plt.xticks(pos_tick, [1] + [""] * (too_close - 1) + orders[too_close:], rotation=0, size=10)
        plt.legend(loc=7)
        plt.savefig("experiment_q_image/Rank.pdf")
        plt.show()

        # plt.figure(figsize=(10, 8))
        label = ["OT"] + ["q = " + str(i) for i in orders] + ["SRW k=1"]
        for k in range(len(orders) + 2):
            t = k / (len(orders) + 1)
            if k in [0, 3, 6, 9, 11]:
                plt.plot(np.arange(1, len(list_eigenvalues[k]) + 1), list_eigenvalues[k], color=np.array([t, 0, 1 - t]),
                         label=label[k])
            else:
                plt.plot(np.arange(1, len(list_eigenvalues[k]) + 1), list_eigenvalues[k], color=np.array([t, 0, 1 - t]))

        plt.xlabel("Eigenvalues")
        plt.ylabel("Magnitude of Eigenvalues")
        plt.legend(loc=1)
        plt.savefig("experiment_q_image/Eigenvalues.pdf")
        plt.show()
