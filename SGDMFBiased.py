
from __future__ import division
import numpy as np
import scipy.sparse as sp
import conf as CF
import Utility as UL




def matrix_factorization(X, P, Q,BU , BI , ave_rate, K):
    mf = CF.MATRIX_FAC()
    SX= X

    Temp = sp.dok_matrix(X.shape)
    for k in range(K):
        # impact of previous features
        print "Computing Temp"
        for i,j,v in zip(SX.row, SX.col, SX.data):
            Temp[i,j] = P[i,:].dot(Q[j,:].T)
        print "Temp Computed"
        #for i,j,v in zip(SX.row, SX.col, SX.data):
        #    Temp[i,j] = P[i,:].dot(Q[j,:].T)
        for iteration in range(mf.NSTEP):
            cnt = 0
            print "Iteration ", iteration
            for i,j,v in zip(SX.row, SX.col, SX.data):
                if cnt % 1000 == 0:
                    print "CNT ", cnt
                cnt += 1
                eij = v - Temp[i,j] - P[i,k]*Q[j,k] - BU[i,0] - BI[j,0] - ave_rate

                ave_rate = ave_rate + mf.ALPHA * ( 2 * eij )
                BU[i,0] = BU[i,0] + mf.ALPHA * ( 2 *  eij - mf.LAMBDA * BU[i,0] )
                BI[j,0] = BI[j,0] + mf.ALPHA * ( 2 * eij  - mf.LAMBDA * BI[j,0] )

                P[i,k] = P[i,k] + mf.ALPHA * (2 * eij * Q[j,k] - mf.LAMBDA * P[i,k])
                Q[j,k] = Q[j,k] + mf.ALPHA * (2 * eij * P[i,k] - mf.LAMBDA * Q[j,k])

    return P,Q,BU,BI,ave_rate

import scipy.io as io

if __name__ == "__main__":
    print "Read ratings"
    X = io.mmread(CF.FILES.RATINGS)
    print "File read."
    # X = sp.coo_matrix([[0, 1, 2, 2, 1],
    #      [0, 1, 2, 0, 1],
    #      [0, 1, 2, 0, 0],
    #      [0, 1, 2, 3, 0],
    #      [0, 1, 2, 4, 0],])

    N,M = X.shape
    mf = CF.MATRIX_FAC()
    K = mf.K

    P = np.zeros((N,K))+0.1
    Q = np.zeros((M,K))+0.1
    BU = np.zeros ((N,1))
    BI = np.zeros ((M,1))
    ave_rate = 0;
    print "Parameters initialized."
    nP, nQ,BU,BI,ave_rate = matrix_factorization(X, P, Q, BU, BI , ave_rate ,  K)

    Whole_BU = BU*np.ones((1,M))
    Whole_BI = np.ones ((N,1)) * BI.T

    RR = nP.dot(nQ.T) + ave_rate + Whole_BI + Whole_BU

    rmse = 0
    for u, i, r in zip(X.row, X.col, X.data):
        rmse += np.power(RR[u,i] - r, 2)
    rmse /= len(X.data)
    rmse = np.sqrt(rmse)

    print "RMSE=", rmse
