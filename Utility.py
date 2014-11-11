'''
Created on Nov 10, 2014

@author: areza_000
'''
from __future__ import division
import scipy.sparse as sp
import numpy as np
import math

def Compute_RMSE (X,Xhat):
    Y = sp.coo_matrix (X)
    
    sum = 0 ;
    counter = 0 
    
    for i,j,data in zip(Y.row,Y.col,Y.data):
        sum += (data - Xhat[i,j]) **2
        counter += 1 
        
    sum /= counter 
    return math.sqrt (sum)