
from __future__ import division
import numpy as np
import scipy.sparse as sp
import conf as CF
import Utility as UL
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
def matrix_factorization(Train_Data, P, Q,BU , BI , ave_rate, K):
    ''' X is a sparse matrix '''
    mf = CF.MATRIX_FAC()
    #SX= sp.coo_matrix(X)

#Temp = sp.lil_matrix(X.shape)


    '''
    for iteration in range(mf.NSTEP):
        for i,j,v in zip(SX.row, SX.col, SX.data):
            for k in range(K):
                eij = v - P[i,:].dot(Q[j,:].T)
                P[i,k] = P[i,k] + mf.ALPHA * (2 * eij * Q[j,k] - mf.LAMBDA * P[i,k])
                Q[j,k] = Q[j,k] + mf.ALPHA * (2 * eij * P[i,k] - mf.LAMBDA * Q[j,k])

    '''

    for iteration in range(mf.NSTEP):
            print "iteration : ", iteration
            for i,j,v,t in Train_Data:
                eij = v - P[i,:].dot(Q[j,:].T) - BU[i,0] - BI[j,0] - ave_rate
                ave_rate = ave_rate + mf.ALPHA * ( 2 * eij )
                BU[i,0] = BU[i,0] + mf.ALPHA * ( 2 *  eij - mf.LAMBDA * BU[i,0] )
                BI[j,0] = BI[j,0] + mf.ALPHA * ( 2 * eij  - mf.LAMBDA * BI[j,0] )
                P[i,:] = P[i,:] + mf.ALPHA * (2 * eij * Q[j,:] - mf.LAMBDA * P[i,:])
                Q[j,:] = Q[j,:] + mf.ALPHA * (2 * eij * P[i,:] - mf.LAMBDA * Q[j,:])
    return P,Q,BU,BI,ave_rate

import scipy.io as io

def Learn (Train_Data,N,M):
    ''''X = [
         [4,2,3,4],
         [3,1,0,2],
         [4,2,4,0],
         [5,4,3,0],
         [2,1,0,3]
        ]

    X= sp.coo_matrix(X)
    Train_Data = []
    for i,j,v in zip(X.row,X.col,X.data):
        Train_Data.append((i,j,v))
    '''
    #N,M = X.shape
    mf = CF.MATRIX_FAC()
    K = mf.K

    P = np.zeros((N,K))+0.1
    Q = np.zeros((M,K))+0.1
    BU = np.zeros ((N,1))
    BI = np.zeros ((M,1))
    ave_rate = 0;
    print "Start to Learn"
    nP, nQ,BU,BI,ave_rate = matrix_factorization(Train_Data, P, Q, BU, BI , ave_rate ,  K)

    np.save("userMat",nP)
    np.save("MovieMat",nQ)
    np.save("UserBias",BU)
    np.save("MovieBias",BI)
    np.save("average",ave_rate)

    #Whole_BU = BU*np.ones((1,N))
    #Whole_BI = np.ones ((M,1)) * BI.T


    #RR = nP.dot(nQ.T) + ave_rate + Whole_BI + Whole_BU
    #print RR

    rmse = 0

    for u, i, r,t in Train_Data:
        rmse += (float(nP[u,:].dot(nQ[i,:].T)+ave_rate+BU[u,0]+BI[i,0]-r))**2
    rmse /= len(Train_Data)
    rmse = np.sqrt(rmse)

    print "RMSE=", rmse
