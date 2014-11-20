'''
Created on Nov 20, 2014

@author: areza_000
'''
import movies_len.input_process as ml
import SGDMFBiased as SGD
import random as rnd
import conf as CF

Train_Data = []
for u,i,r,t in ml.get_ratings("ml-100/ml-100k/u.data"):
    Train_Data.append((u,i,r))
    
rnd.shuffle(Train_Data)

SGD.Learn(Train_Data, CF.MOVIES_LEN.TRAINING.USERS,CF.MOVIES_LEN.TRAINING.MOVIES)

