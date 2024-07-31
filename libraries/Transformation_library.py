import numpy as np
def hierarchy_tree(Z, dY): #each line of clusters is elements of new clusters obtained by hierarchy_tree
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
def data_transformation(X_, Z, clustersize, clusters, dY):  # pivot log ratios
    stats_train = np.exp(np.mean(np.log(X_), axis=1, keepdims=True)) # geometric mean as the first feature
    for i in range(dY - 2, -1, -1):
        cl0, cl1 = int(Z[i, 0]), int(Z[i, 1])
        s0, s1 = int(clustersize[cl0]), int(clustersize[cl1])
        cluster0, cluster1 = clusters[cl0, 0:s0].astype(int), clusters[cl1, 0:s1].astype(int)
        mean0, mean1 = np.mean(np.log(X_[:, cluster0]), axis=1), np.mean(np.log(X_[:, cluster1]), axis=1)
        stats_train = np.append(stats_train, np.exp(mean0 - mean1).reshape(-1, 1), axis=1)
    return stats_train
def CLR(X, keepdim = True): #Centered Log-Ratio
    GeomeanX = np.exp(np.mean(np.log(X), axis = -1, keepdims=True))
    clrX = np.log(np.divide(X,  GeomeanX))
    if keepdim == False:
        clrX = np.delete(clrX, -1, -1)
    return clrX
def softmax(Y): #inverse of CLR
    n = np.shape(Y)[0]
    m = np.shape(Y)[1]
    U = np.zeros((n,m+1))
    U[:,:-1] = Y
    U[:,-1] = - np.sum(Y,axis=1)
    exp = np.exp(U - np.tile(np.max(U,axis = 1).reshape(-1,1),(1,np.shape(U)[1]))) #trick to avoid overflow
    return exp / np.sum(exp, axis=-1, keepdims=True)