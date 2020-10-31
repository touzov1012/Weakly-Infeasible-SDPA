# -*- coding: utf-8 -*-
"""
A collection of helper functions used by any algorithm for extending
SDE to a general weak SDP and messing up structure using elementary operations
and rotations.

@author: alex
"""

import wsdp_utility as util
import numpy as np


def ExtendWeakSDE(A, b, X, count, v = 1):
    """
    Extend a weak SDE to larger size by selecting symmetric matrices orthogonal
    to X, v is the certificate scaling factor such that AX = vb
    """
    
    n = A.shape[1]
    
    truInd = np.triu_indices(n)
    trlInd = np.tril_indices(n, k=-1)
    
    Xc = X.copy()
    Xc[:,np.arange(0,n),np.arange(0,n)] = 0
    Xc += X
    
    Xvec = Xc[:-1,truInd[0],truInd[1]]
    
    for i in range(0,count):
        
        avec = util.NullSpaceInt(Xvec)
        
        Anew = np.zeros((n,n), dtype=int)
        Anew[truInd] = avec
        Anew[trlInd] = Anew.T[trlInd]
        
        bnew = int(np.tensordot(Anew, X[-1,:,:], axes=((0,1),(0,1))))
        
        Anew *= np.abs(v)
        
        b = np.append(b, bnew)
        A = np.append(A, [Anew], axis=0)
    
    return [A, b]
        

def RotateBases(A, X, steps):
    """
    Rotate each of the a_i and x_j using a random rotation controlled by "steps"
    """
    
    [A[0,:,:], T, Ti] = util.Rotate(A[0,:,:], steps)
    
    for i in range(1,A.shape[0]):
        A[i,:,:] = np.matmul(np.matmul(T.T, A[i,:,:]), T)
    
    for i in range(0,X.shape[0]):
        X[i,:,:] = np.matmul(np.matmul(Ti, X[i,:,:]), Ti.T)
        
    return [T, Ti]
    

def RotateSequence(A,b,steps=40):
    """
    Apply elementary row operations to mess up an instance (A,b)
    """
    
    n = A.shape[0]
    
    F = np.eye(n,dtype=int)
    [F,M,Mi] = util.Rotate(F,steps)
    F = np.matmul(F, Mi)
    [F,M,Mi] = util.Rotate(F,steps)
    F = np.matmul(Mi.T, F)
    
    A = np.tensordot(F,A,axes=(1,0))
    b = np.matmul(F,b)
    
    return [A, b, F]

    
    
    