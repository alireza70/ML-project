'''
Created on Nov 9, 2014

@author: areza_000
'''

from __future__ import division
import numpy as np
import conf as CF
import scipy.sparse as sp
import math
import scipy.io as io
import types 
import Utility as UL

base = CF.BASE_NEIGHBOURHOOD()

#X=sp.csc_matrix((17000,500000))

#Sim_Matrix is a row sparse matrix and Sim[i,j] is 1 if j is similar to i (Top k similars)

def Update_Weights (X,rui,bu,bi,ave_rate,u,i,Sim_Matrix,B_Predictor,Movie_Cor):
    
    #print "begin of section 4" , u , i, X[u,:].shape , Sim_Matrix[i,:].shape,B_Predictor[u,:].shape
    tempx = X[u,:].multiply(Sim_Matrix[i,:])
    tempb = B_Predictor[u,:].multiply(Sim_Matrix[i,:])
    
    lentgh = len(sp.coo_matrix(tempx).data)
    
    temp = (1/math.sqrt(lentgh)) *(tempb - tempx).dot(Movie_Cor[i,:].T).todense()
    
    #TODO
    #print "bug: ", type(bu) , type(bi) , type(temp), temp.shape
    
    error = (rui - ave_rate - bu - bi - float(temp))
    
    bu = bu + base.ALPHA*(error - base.LAMBDA*bu)
    bi= bi + base.ALPHA*(error - base.LAMBDA*bi)
    
    ave_rate = ave_rate + base.ALPHA*(error)
    
    nonzeros = sp.coo_matrix(tempx)
    
    for i,j,data in zip(nonzeros.row,nonzeros.col,nonzeros.data):
        Movie_Cor[i,j] = Movie_Cor[i,j] + base.ALPHA * ( (1/math.sqrt(lentgh)) * error * 
                                                          (X[u,i]-B_Predictor[u,i]) - base.LAMBDA*Movie_Cor[i,j] )
    
    return bu,bi,ave_rate      


def Compute_Coef (a,b):
    
    #print "1","a =", a , "b = ", b
    Sa= sp.coo_matrix(a)
    Sb = sp.coo_matrix(b)
    
    common =  [filter(lambda x: x in Sa.row, Sb.row)][0]
    
    print "common! = ", common , type(common)
    
    #common = np.squeeze(np.asarray(common))
    
    #for efficiency we can use coo_matrix to compute common_of_a or common_of_b
    
    if (len(common) == 0):
        return 0 
    
    common_of_a = [a[x,0] for x in common]
    common_of_b = [b[x,0] for x in common]
    
    
    c= [i*j for i,j in zip(common_of_a,common_of_b)]
    covariance = np.mean(c)-np.mean(common_of_a)*np.mean(common_of_b)
    #print "cov = ", covariance
    #a = np.squeeze(np.asarray(X))
    #b = np.squeeze(np.asarray(Y))
    
    res = ( len(common_of_a) / (len(common_of_a) + base.LAMBDA2) ) * ( covariance / (np.sqrt( np.var(common_of_a) * np.var(common_of_b) ) ) )
    #print res
    #print "2 ", a , b 
    return res 


def Compute_Neighbours (X,i,m,k):
    temp = X[:,i]
    
    res = [Compute_Coef(temp,X[:,j]) for j in range(m)]
    
    res = np.asarray(res)
    #print "enf of section 1"
    return np.argpartition(res, -k)[-k:]

def Make_Weight_Matrix(X,m,u,k):
    
    res = sp.lil_matrix((m,m))
    
    for i in range(m):
        temp = Compute_Neighbours(X, i, m, k)
        #print i,temp
        for j in range (k):
            res[i,j] = 1
    
    #print "end of section 2"
    return res ; 
    

def Learn_Algorithm(X,B_predictor,u,m):
    
    Sim_Matrix = Make_Weight_Matrix(X, m, u, base.K)
    Movie_Cor = sp.lil_matrix((m,m))
    BU = sp.lil_matrix((u,1))
    BI = sp.lil_matrix((m,1))
    ave_rate = 0 ;
    
    SX = sp.coo_matrix(X)
    print "end of section 3"
    for i in range(base.NSTEP):
        print "i = ", i
        for x,y,data in zip(SX.row,SX.col,SX.data):
            #print x , y 
            BU[x,0],BI[y,0],ave_rate = Update_Weights(X,data , BU[x,0], BI[y,0], ave_rate, x, y, Sim_Matrix, B_predictor, Movie_Cor)
    
    '''io.mmwrite("Ave",ave_rate) 
    io.mmwrite("BU", BU)
    io.mmwrite("BI",BI)
    io.mmwrite("Sim_Matrix",Sim_Matrix)
    io.mmwrite("Movie_Cor",Movie_Cor)'''
    
    print "end of section 5"        
            
    
    return ave_rate,BU,BI,Movie_Cor,Sim_Matrix

def Predict(X,ave_rate, u,i,BU,BI,Movie_Cor,Sim_Matrix,B_Predictor):
    
    print "bug : ", u , i , BU.shape, BI.shape
    print BU[u,0] , BI[i,0]
    res = ave_rate + BU[u,0] + BI[i,0] ;
    print "kir"
    tempx = X[u,:].multiply(Sim_Matrix[i,:])
    tempb = B_Predictor[u,:].multiply(Sim_Matrix[i,:])
  
    lentgh = len(sp.coo_matrix(tempx).data)
    
    temp = (1/math.sqrt(lentgh)) *(tempb - tempx).dot(Movie_Cor[i,:].T).todense()
    
    res = res +  float(temp) 
    print "kir2"
    return res 

def Run_For_Test ():
    X = [
         [4,2,3,4],
         [3,1,0,2],
         [4,2,4,0],
         [5,4,3,0],
         [2,1,0,3],
        ]

    X= sp.csr_matrix(X)
    u,m = X.shape
    
    B_Predictor = [[ 3.84451398 , 2.2492453 ,  2.88938797 , 4.03037924],
                   [ 2.31250009 , 1.34276181 , 1.74429281 , 2.42130592],
                   [ 4.26895323 , 2.48925027 , 3.21353468 , 4.47289195],
                   [ 4.98630911 , 2.94694872 , 3.72910929 , 5.23611119],
                   [ 2.32808477 , 1.35854455 , 1.7518738 ,  2.43960498]]
    
    B_Predictor = sp.csr_matrix(B_Predictor)
    ave_rate,BU,BI,Movie_Cor,Sim_Matrix = Learn_Algorithm(X, B_Predictor, u, m)
    print "u , m", u , m
    
    print BU.shape, BI.shape
    res = np.zeros((u,m))
    print "res" ,res.shape
    for i in range (u):
        for j in range (m):
            res[i,j] = Predict(X, ave_rate, i, j, BU, BI, Movie_Cor, Sim_Matrix, B_Predictor)
    
    print res
    print BU.T.todense() , BI.T.todense()
    print ave_rate
    print UL.Compute_RMSE(X, res)
    
if __name__ == "__main__":
    #X = np.matrix([[1,2,3],[1,2,2],[3,4,5],[3,5,5]]).T
    #X = sp.csc_matrix(X) 
    #print Make_Weight_Matrix(X, 4, 3, 2)
    a = [1 ,2 , 3]
    b = [4 , 5 ,6]
    common =  [filter(lambda x: x in a, b)]
    X=np.squeeze([[2]])
    #if ( not isinstance(X, list)):
    #    print list(X), X
    
    Run_For_Test()

            

    



