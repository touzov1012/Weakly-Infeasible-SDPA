# -*- coding: utf-8 -*-
"""
The bilinear system algorithm. Simple and fast way of generating any
weak SDP given the structure of the SDE forms for the SDP and its certificate.

@author: alex
"""

import wsdp_utility as util
import numpy as np


def CreateWeakSysCertSDE(P, Q, val_rng, plus_rng):
    """
    Create a weakly infeasible SDP and its certificate using the bilinear alg.
    """
    
    if not util.ValidateCertStructure(P, Q):
        util.LogError("Not a valid P,Q structure")
        return None
    
    k = np.amax(P) - 1
    l = np.amax(Q) - 1
    
    A = util.CreateSDE(val_rng, plus_rng, P)
    X = util.CreateSDE(val_rng, plus_rng, Q)
    
    # a map from the pair of matrices to the index which controls the product
    map_col = util.GetPQCoord(P,Q,0,0)[1]
    map_rows = [util.GetPQCoord(P,Q,x,0)[0] for x in range(0,k)]
    
    X[:,map_rows,map_col] = 0
    X[:,map_col,map_rows] = 0
    
    B = np.zeros((k,l), dtype=int)
    B[k-1,l-1]=-1
    
    for r in range(0,k):
        B[r,:] = np.array([B[r,c] - int(np.tensordot(A[r+1,:,:], X[c+1,:,:], axes=((0,1),(0,1)))) for c in range(0,l)])
        
        if not any(B[r,:]):
            # this can also be skipped in favor of "else" in the general case
            A[r+1,map_rows[r],map_col] = 0
            X[1:l+1,map_rows[r],map_col] = np.random.randint(val_rng[0], val_rng[1], size=(l,))
        else:
            A[r+1,map_rows[r],map_col] = np.random.choice(util.Divisors(B[r,:]))
            # we should divide by 2 here; however, such a value may not be integral
            X[1:l+1,map_rows[r],map_col] = B[r,:] / A[r+1,map_rows[r],map_col]
    
    
    # apply symmetry to our matrices
    A[:,map_col,map_rows] = A[:,map_rows,map_col]
    X[:,map_col,map_rows] = X[:,map_rows,map_col]
    
    # the above code produces A which is scaled 2 times as large in some of the
    # off diagonal elements. Thus, to stay integer valued, we rescale the remaining
    # elements of A by 2
    Afilt = np.ones_like(A, dtype=int) * 2
    Afilt[:,map_rows,map_col] = 1
    Afilt[:,map_col,map_rows] = 1
    
    
    A = np.multiply(A, Afilt)
    
    b = np.zeros((k+1,), dtype=int)
    b[k] = -2
    
    return [A, b, X]


    
    
    