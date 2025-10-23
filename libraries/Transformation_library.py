"""
This library is the colelction of methods for transformation. It includes the methods for log-transfornations and inverse functions.
"""
import numpy as np
def hierarchy_tree(Z, dY):
    """
    The function constructs clusters from hierarchical tree.
    """
    clusters = -1 * np.ones((2 * dY - 1, dY))
    clusters[0:dY, 0] = np.arange(dY)
    clustersize = np.zeros(2 * dY - 1)
    clustersize[0:dY] = np.ones(dY)
    for i in range(0, dY - 1):
        cl0, cl1 = int(Z[i, 0]), int(Z[i, 1])
        s0, s1 = int(clustersize[cl0]), int(clustersize[cl1])
        new_cluster = np.concatenate([clusters[cl0, 0:s0], clusters[cl1, 0:s1]])
        clustersize[dY + i] = Z[i, 3]
        clusters[dY + i, 0:int(clustersize[dY + i])] = new_cluster
    return clustersize, clusters
def data_transformation(X_, Z, clustersize, clusters, dY):
    """
    Data transformation by pivot log ratios
    """
    stats = np.exp(np.mean(np.log(X_), axis=1, keepdims=True)) # geometric mean as the first feature
    for i in range(dY - 2, -1, -1):
        cl0, cl1 = int(Z[i, 0]), int(Z[i, 1])
        s0, s1 = int(clustersize[cl0]), int(clustersize[cl1])
        cluster0, cluster1 = clusters[cl0, 0:s0].astype(int), clusters[cl1, 0:s1].astype(int)
        mean0, mean1 = np.mean(np.log(X_[:, cluster0]), axis=1), np.mean(np.log(X_[:, cluster1]), axis=1)
        stats = np.append(stats, np.exp(mean0 - mean1).reshape(-1, 1), axis=1)
    return stats
def CLR(X, keepdim = True): #Centered Log-Ratio
    """
    Data transformation by centered log ratios
    """
    GeomeanX = np.exp(np.mean(np.log(X), axis = -1, keepdims=True))
    clrX = np.log(np.divide(X,  GeomeanX))
    if keepdim == False:
        clrX = np.delete(clrX, -1, -1)
    return clrX
def softmax(Y): #inverse of CLR
    """
    Softmax function, that is inverse of CLR
    """
    n = np.shape(Y)[0]
    m = np.shape(Y)[1]
    U = np.zeros((n,m+1))
    U[:,:-1] = Y
    U[:,-1] = - np.sum(Y,axis=1)
    exp = np.exp(U - np.tile(np.max(U,axis = 1).reshape(-1,1),(1,np.shape(U)[1]))) #trick to avoid overflow
    return exp / np.sum(exp, axis=-1, keepdims=True)
def LR(X):
    """
    Data transformation by all log ratios
    """
    n = np.shape(X)[0]
    d = np.shape(X)[1]
    Y = np.zeros((n, int(d*(d-1)/2)))
    k = 0
    for i in range(1,d):
        for j in range(0,i):
            Y[:,k]=np.log(X[:,i]) - np.log(X[:,j])
            k+=1
    return Y

def transform_2(X_, arg_bacteria):
    """
    Log transformation given only two features. The number of bacteria is given in argument arg_bacteria.
    """
    stats = np.divide(X_[:, 1, arg_bacteria[0]], X_[:, 1, arg_bacteria[1]]).reshape(-1,1)
    stats = np.append(stats, np.divide(X_[:, 0, arg_bacteria[0]], X_[:, 0, arg_bacteria[1]]).reshape(-1,1), axis=1)
    return stats