'''
Created on Nov 10, 2014

@author: areza_000
'''


from __future__ import division
import copy
try:
    import numpy as np
    import scipy.sparse as sp
except:
    print "This implementation requires the numpy and scipy modules."
    exit(0)
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
def matrix_factorization(X, P, Q,BU , BI , ave_rate, K):
    ''' X is a sparse matrix '''
    mf = CF.MATRIX_FAC()
    SX= sp.coo_matrix(X)
    
    Temp = sp.lil_matrix(X.shape)
    print Temp
    #ST = sp.coo_matrix(Temp)
    
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
        Temp = P.dot(Q.T)
        #for i,j,v in zip(SX.row, SX.col, SX.data):
        #    Temp[i,j] = P[i,:].dot(Q[j,:].T)
        print Temp
        
        for iteration in range(mf.NSTEP):
            
            for i,j,v in zip(SX.row, SX.col, SX.data):
                eij = v - Temp[i,j] - P[i,k]*Q[j,k] - BU[i,0] - BI[j,0] - ave_rate
                
                ave_rate = ave_rate + mf.ALPHA * ( 2 * eij )
                BU[i,0] = BU[i,0] + mf.ALPHA * ( 2 *  eij - mf.LAMBDA * BU[i,0] )  
                BI[j,0] = BI[j,0] + mf.ALPHA * ( 2 * eij  - mf.LAMBDA * BI[j,0] )
                
                P[i,k] = P[i,k] + mf.ALPHA * (2 * eij * Q[j,k] - mf.LAMBDA * P[i,k])
                Q[j,k] = Q[j,k] + mf.ALPHA * (2 * eij * P[i,k] - mf.LAMBDA * Q[j,k])
                 
    return P,Q,BU,BI,ave_rate

###############################################################################

if __name__ == "__main__":
    X = [
         [4,2,3,4],
         [3,1,0,2],
         [4,2,4,0],
         [5,4,3,0],
         [2,1,0,3]
        ]

    X= sp.csc_matrix(X)

    N,M = X.shape
    mf = CF.MATRIX_FAC()
    K = mf.K

    P = np.zeros((N,K))+0.1
    Q = np.zeros((M,K))+0.1
    BU = np.zeros ((N,1))
    BI = np.zeros ((M,1))
    ave_rate = 0;
    
    nP, nQ,BU,BI,ave_rate = matrix_factorization(X, P, Q, BU, BI , ave_rate ,  K)
    
    Whole_BU = BU*np.ones((1,M))
    Whole_BI = np.ones ((N,1)) * BI.T 
    
    #print nP.dot(nQ.T) + ave_rate + Whole_BI + Whole_BU 

    #print ave_rate , BU , BI
    
    #print UL.Compute_RMSE(X, nP.dot(nQ.T) + ave_rate + Whole_BI + Whole_BU) 