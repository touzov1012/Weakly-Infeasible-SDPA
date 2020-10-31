# -*- coding: utf-8 -*-
"""
A series of utility and helper functions for algorithms

@author: alex
"""

import numpy as np


def LogError(error):
    print("ERROR: " + str(error))
    

def LogWarning(warning):
    print("WARNING: " + str(warning))


"""
Helper methods for applying invertible rotations to A to create A'
Invertible column operations are stored in a matrix T
Invertible row operations are stored in a matrix S
The final transformation is of the form A' = SAT
"""
def Add_c2c(A,T,Ti,i,j,c):
    A[:,j] += A[:,i] * c
    T[:,j] += T[:,i] * c
    Ti[i,:] -= Ti[j,:] * c
        
def Add_r2r(A,S,Si,i,j,c):
    A[j,:] += A[i,:] * c
    S[j,:] += S[i,:] * c
    Si[:,i] -= Si[:,j] * c
    
def Swp_col(A,T,Ti,i,j):
    A[:,i], A[:,j] = A[:,j], A[:,i].copy()
    T[:,i], T[:,j] = T[:,j], T[:,i].copy()
    Ti[i,:], Ti[j,:] = Ti[j,:], Ti[i,:].copy()
    
def Swp_row(A,S,Si,i,j):
    A[i,:], A[j,:] = A[j,:], A[i,:].copy()
    S[i,:], S[j,:] = S[j,:], S[i,:].copy()
    Si[:,i], Si[:,j] = Si[:,j], Si[:,i].copy()
    
def Flp_col(A,T,Ti,i):
    A[:,i] *= -1
    T[:,i] *= -1
    Ti[i,:] *= -1
        
def Flp_row(A,S,Si,i):
    A[i,:] *= -1
    S[i,:] *= -1
    Si[:,i] *= -1
"""
Elementary operations end
"""

def Rotate(X, steps):
    """
    Apply an arbitary rotation to X invertible over integers so X' = T^tXT
    """
    
    n = X.shape[0]
    
    A = X.copy()
    T = np.eye(n, dtype=int)
    Ti = np.eye(n, dtype=int)
    
    for i in range(0,steps):
        op = np.random.randint(0,3)
        if op == 0:
            Flp_col(A, T, Ti, np.random.randint(0,n))
        elif op == 1:
            Swp_col(A, T, Ti, np.random.randint(0,n), np.random.randint(0,n))
        else:
            c1 = np.random.randint(0,n)
            c2 = np.random.randint(0,n)
            if c1 == c2:
                continue
            Add_c2c(A, T, Ti, c1, c2, 1)
    
    A = np.matmul(T.T, A)
    
    return [A, T, Ti]


def Condition(A):
    """
    Get the condition number of A, sequence of sym matrices by vectorizing
    the upper triangular part of each A_i ans stacking the horizontal vectors
    vertically to obtain a matrix from which the condition number is taken
    """
    
    n = A.shape[1]
    
    truInd = np.triu_indices(n)
    
    Avec = A[:,truInd[0],truInd[1]]
    
    return np.linalg.cond(Avec)


def NullSpaceInt(X, T = None):
    """
    Solve a system of linear diophantine equations for an int random sample
    from the null space of int X, T is column transformation to get X into
    Smith form, if T is passed in, then X is assumed to already be in Smith
    form
    """
    
    n = X.shape[1]
    
    if T is None:
        [A, S, Si, T, Ti] = Smith(X)
    else:
        A = X
    
    seed = np.random.randint(-2, 3, size=n)
    clip = np.where(A.any(axis=0))[0]
    seed[clip] = 0
    
    return np.matmul(T, seed)
    

def Smith(X, S = None, Si = None, T = None, Ti = None, index = 0, clone = True):
    """
    Calculate smith decomposition along with basis transformation matrices
    to make SXT = A diagonal
    """
    
    m = X.shape[0]
    n = X.shape[1]
    
    A = X
    if clone:
        A = X.copy()
    
    if S is None:
        S = np.eye(m, dtype=int)
    if Si is None:
        Si = np.eye(m, dtype=int)
    if T is None:
        T = np.eye(n, dtype=int)
    if Ti is None:
        Ti = np.eye(n, dtype=int)
    
    def add_c2c(i,j,c):
        Add_c2c(A, T, Ti, i, j, c)
        
    def add_r2r(i,j,c):
        Add_r2r(A, S, Si, i, j, c)
    
    def col_swp(i,j):
        Swp_col(A, T, Ti, i, j)
    
    def row_swp(i,j):
        Swp_row(A, S, Si, i, j)
    
    def flp_col(i):
        Flp_col(A, T, Ti, i)
        
    def flp_row(i):
        Flp_row(A, S, Si, i)
        
    def col_fixed(i):
        col = A[:,i]
        return not col[np.arange(len(col))!=i].any()
    
    def row_fixed(i):
        row = A[i,:]
        return not row[np.arange(len(row))!=i].any()
    
    def min_nonzero_col(i):
        col = A[:,i]
        val = np.min(col[np.nonzero(np.abs(col))])
        ind = np.where(np.logical_or(col == val, col == -val))[0][0]
        return [ind, col[ind]]
    
    def min_nonzero_row(i):
        row = A[i,:]
        val = np.min(row[np.nonzero(np.abs(row))])
        ind = np.where(np.logical_or(row == val, row == -val))[0][0]
        return [ind, row[ind]]
    
    
    # end of recursion, return result
    if index >= min(m, n):
        return [A, S, Si, T, Ti]
    
    while not row_fixed(index) or not col_fixed(index):
        # fix row
        while not row_fixed(index):
            [ind, val] = min_nonzero_row(index)
            
            if ind != index:
                col_swp(ind, index)
            
            if A[index, index] < 0:
                flp_col(index)
                
            for i in range(0, n):
                if i == index:
                    continue
                
                factor = A[index,i] // A[index,index]
                add_c2c(index, i, -factor)
        
        # fix col
        while not col_fixed(index):
            [ind, val] = min_nonzero_col(index)
            
            if ind != index:
                row_swp(ind, index)
            
            if A[index, index] < 0:
                flp_row(index)
                
            for i in range(0, m):
                if i == index:
                    continue
                
                factor = A[i,index] // A[index,index]
                add_r2r(index, i, -factor)
        
    
    return Smith(A, S, Si, T, Ti, index + 1, False)


def Divisors(x):
    """
    find all divisors of a set of numbers x
    """
    
    y = np.abs(x)
    M = np.amax(y)
    
    if M == 0:
        return [1]
    
    ds = set(range(1,M+1))
    n = len(y)
    
    for i in range(0,n):
        if len(ds) == 1:
            return [1]
        
        m = 1
        for j in range(2,M+1):
            if x[i] % j != 0:
                ds.discard(j)
            else:
                m = j
        M = m
    
    return list(ds)
    

def Pid2Pvec(Pid):
    """
    Converts an array of index ids (0,1,...,k) to a list where list indices
    map to binary vectors identifying that index.
    Ex:
        Pid2Pvec([0,1,3]):
            [[True, False, False],
             [False, True, False],
             [False, False, False],
             [False, False, True]]
    """
    
    k = np.amax(Pid)
    
    Pvec = []
    
    for i in range(0,k+1):
        Pvec.append([x == i for x in Pid])
    
    return Pvec


def Pvec2Pid(Pvec):
    """
    This is the inverse map to the function Pid2Pvec() above.
    """
    
    k = len(Pvec)
    n = len(Pvec[0])
    
    Pid = np.zeros((n,), dtype=int)
    
    for i in range(0, k):
        level = np.array(Pvec[i]) * i
        Pid += level
    
    return list(Pid)


def ValidatePid(Pid):
    """
    Check if the input denotes a valid index partition from an id vector
    """
    
    if np.amin(Pid) != 0:
        LogError("Pids must start from 0")
        return False
    
    k = np.amax(Pid)
    n = len(Pid)
    
    check = np.zeros((k+1,), dtype=int)
    
    for i in range(0,n):
        check[Pid[i]] = 1
    
    if np.sum(check) != k+1:
        LogError("Not all ids in the range of Pid were used")
        return False
    
    return True


def ValidatePvec(Pvec):
    """
    Check if input partition vectors are a valid partition
    """
    
    mat = np.array(Pvec)
    flat0 = np.sum(mat, axis=0)
    flat1 = np.sum(mat, axis=1)
    
    if any(x != 1 for x in flat0):
        LogError("Pvec does not form a partition")
        return False
    
    if any(x == 0 for x in flat1):
        LogError("Pvec does not use all indices")
        return False
    
    return True


def ValidateWeakCert(A,b,X,v,inspect=False):
    """
    Check if X is a weak certificate to the system (A,b) such that Ax_l=vb
    """
    
    B = np.tensordot(A,X, axes=((1,2),(1,2)))
    
    if inspect:
        print(B)
    
    if B[:,:-1].any():
        print("certificate incorrect for one of first k terms")
        return False
    
    if not np.equal(B[:,-1],b*v).all():
        print("certificate incorrect for k+1 term")
        return False
    
    return True


def ValidateCertStructure(P, Q):
    """
    Check if Q is a valid certificate structure for SDE structure P
    """
    
    if not ValidatePid(P):
        LogError("P is not a valid SDE structure")
        return False
    
    if not ValidatePid(Q):
        LogError("Q is not a valid SDE structure")
        return False
    
    if(len(P) != len(Q)):
        LogError("P and Q do not match dimensions")
        return False
    
    k = np.amax(P)
    l = np.amax(Q)
    
    p1 = np.array([x == 0 for x in P])
    pk = np.array([x != k for x in P])
    q1 = np.array([x == 0 for x in Q])
    ql = np.array([x != l for x in Q])
    
    if any(p1 & ql):
        LogError("P1 intersects Q")
        return False
    
    if any(q1 & pk):
        LogError("Q1 intersects P")
        return False
    
    return True


def FillRandColSym(A, c, a_range):
    """
    Fill the row and column of a symmetric matrix with random values from
    the range [amin, amax]
    """
    n = np.size(A,0)
    for i in range(0,n):
        A[c,i] = np.random.randint(a_range[0], a_range[1] + 1)
        A[i,c] = A[c,i]


def GetPQCoord(P, Q, p, q):
    """
    Get a coordinate from the block (P_p, Q_q)
    """
    
    r = np.flatnonzero([x == p for x in P])[0]
    c = np.flatnonzero([x == q for x in Q])[0]
    
    return (r,c)
    

def CreateSDE(A_range, plus_range, Pid):
    """
    Create a random SDE with arbitrary elements in the range [A_min, A_max]
    and positive blocks with range [plus_min, plus_max] and structure
    definied by Pid
    """
    
    k = np.amax(Pid)
    n = len(Pid)
    
    A = []
    
    for i in range(0,k):
        
        a_i = np.zeros((n,n), dtype=int)
        
        for j in range(0,n):
            
            if Pid[j] > i:
                continue
            elif Pid[j] == i:
                a_i[j,j] = np.random.randint(plus_range[0], plus_range[1] + 1)
            else:
                FillRandColSym(a_i, j, A_range)
        
        A.append(a_i)
    
    return np.reshape(A, (len(A), n, n))


def CreateSDEbvec(k):
    """
    Helper method to create a vector (0,0,...,0,-1)
    """
    
    b = np.zeros((k,), dtype=int)
    b[-1] = -1
    
    return b


def CreateConsecPartition(sizes):
    """
    Create a consecutive index partition of 1:sum(sizes)
    """
    
    n = np.sum(sizes)
    k = len(sizes)
    
    P = np.zeros((n,), dtype=int)
    cnt = 0
    for i in range(0,k):
        for j in range(0,sizes[i]):
            P[cnt] = i
            cnt += 1
        
    return list(P)


def CreateCertificateStructure(parts, Pid):
    """
    Create a random valid structure with "parts" partitions for a certificate
    of a SDE with structure given by Pid
    """
    
    if parts < 3:
        LogError("partition size should be >= 3")
        return None
    
    n = len(Pid)
    k = np.amax(Pid)
    
    Q = np.full((n,), parts - 1)
    
    fillOrder = np.random.permutation(n)
    
    # make sure that at least one element of the last structure block of Pid
    # corresponds to the first structure block of Q
    pk = np.flatnonzero([x == k for x in Pid])
    start = np.random.choice(pk)
    Q[pk] = 0
    
    cnt = 1
    for i in fillOrder:
        
        # skip this index if it belongs to the first structure block
        # or to the index we choose for the new first structure element
        if Pid[i] == 0 or i == start:
            continue
        
        # make sure each partition has at least one index
        if cnt < parts - 1:
            Q[i] = cnt
            cnt += 1
        else:
            Q[i] = np.random.randint(1, parts)
        
    if cnt != parts - 1:
        LogWarning("partition of size " + str(parts) + " is too large")
        Q = [cnt if x == parts - 1 else x for x in Q]
    
    return list(Q)
    

def ConvertSDE2IdMap(Q, diagOnly=True):
    """
    Replace values of free blocks with "1", positive definite block elements "2"
    """
    
    k = np.amax(Q)
    n = len(Q)
    
    idMap = np.zeros(shape=(k,n,n), dtype=int)
    for i in range(0,k):
        Xind = np.array(Q) < i
        Pind = np.array(Q) == i
        if diagOnly:
            idMap[i,Pind,Pind] = 2
        else:
            idMap[i,:,:][np.ix_(Pind,Pind)] = 2
        idMap[i,Xind,:] = 1
        idMap[i,:,Xind] = 1
    
    return idMap

















    
 