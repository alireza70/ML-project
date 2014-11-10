'''
Created on Nov 6, 2014

@author: areza_000
'''

#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
import copy
try:
    import numpy as np
    import scipy.sparse as sp
except:
    print "This implementation requires the numpy and scipy modules."
    exit(0)
import conf as CF
###############################################################################

"""
@INPUT:
    X     : a matrix to be factorized, dimension U x M
    P     : an initial matrix of dimension U x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(X, P, Q, K):
    ''' X is a sparse matrix '''
    mf = CF.MATRIX_FAC()
    SX= sp.coo_matrix(X)

    for iteration in range(mf.NSTEP):
        for k in range(K):
            for i,j,v in zip(SX.row, SX.col, SX.data):
                eij = v -P[i,:].dot(Q[:,j])
                P[i,k] = P[i,k] + mf.ALPHA * (2 * eij * Q[k][j] - mf.LAMBDA * P[i,k])
                Q[k,j] = Q[k,j] + mf.ALPHA * (2 * eij * P[i][k] - mf.LAMBDA * Q[k,j])
    Temp = copy.deepcopy(X)
    ST = sp.coo_matrix(Temp)

    '''
    for iteration in range(mf.NSTEP):
        for i,j,v in zip(SX.row, SX.col, SX.data):
            for k in range(K):
                eij = v - P[i,:].dot(Q[j,:].T)
                P[i,k] = P[i,k] + mf.ALPHA * (2 * eij * Q[j,k] - mf.LAMBDA * P[i,k])
                Q[j,k] = Q[j,k] + mf.ALPHA * (2 * eij * P[i,k] - mf.LAMBDA * Q[j,k])

    '''
    for k in range(K):
        # impact of previous features
        for i,j,v in zip(ST.row, ST.col, ST.data):
            Temp[i,j] = P[i,:].dot(Q[j,:].T)

        for iteration in range(mf.NSTEP):

            for i,j,v in zip(SX.row, SX.col, SX.data):
                eij = v - Temp[i,j] - P[i,k]*Q[j,k]

                P[i,k] = P[i,k] + mf.ALPHA * (2 * eij * Q[j,k] - mf.LAMBDA * P[i,k])
                Q[j,k] = Q[j,k] + mf.ALPHA * (2 * eij * P[i,k] - mf.LAMBDA * Q[j,k])

    return P,Q

###############################################################################

if __name__ == "__main__":
    X = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    X= sp.csr_matrix(X)

    N,M = X.shape
    K = 2

    P = np.zeros((N,K))+0.1
    Q = np.zeros((M,K))+0.1

    nP, nQ = matrix_factorization(X, P, Q, K)
    print nP.dot(nQ.T)
