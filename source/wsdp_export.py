# -*- coding: utf-8 -*-
"""
A collection of functions related to output and visualization of results
produced by algorithms.

@author: alex
"""

import os
import numpy as np
import scipy.io as sio
from scipy import sparse
from PIL import Image, ImageDraw


def CreateSDEMatrixImage(Q, index, A = None, blockSize = 24, lineSize = 2, useColor = True):
    """
    Create an image of a matrix element of a given SDE structure
    """
    
    k = np.amax(Q)
    structX = np.logical_and(np.array(Q) < index, np.repeat(index < k, len(Q)))
    structI = np.logical_and(np.array(Q) == index, np.repeat(index < k, len(Q)))
    
    n = len(Q)
    sz = n * blockSize + (n+1) * lineSize
    
    # look up table for each color of our array
    # 0 - background
    # 1 - lines
    # 2 - X
    # 3 - +
    LUT = np.array([[255,255,255],[90,100,100],[0,150,255],[255,81,0]], dtype=np.uint8)
    if not useColor:
        LUT = np.array([[255,255,255],[90,100,100],[255,255,255],[255,255,255]], dtype=np.uint8)
    
    # draw the base grid as a numpy array to make pixel perfect
    grid = np.zeros((sz,sz), dtype=np.uint8)
    
    for i in range(0,n):
        x = lineSize + i * (blockSize + lineSize)
        if structX[i]:
            grid[x:x+blockSize,:] = 2
            grid[:,x:x+blockSize] = 2
        for j in range(0,n):
            y = lineSize + j * (blockSize + lineSize)
            if structI[i] and structI[j]:
                grid[x:x+blockSize,y:y+blockSize] = 3
    
    for i in range(0,n+1):
        x =  i * blockSize + i * lineSize
        grid[x:x+lineSize,:] = 1
        grid[:,x:x+lineSize] = 1
    
    # color the pixels based on the lookup table provided
    pxls = LUT[grid]
    
    im = Image.fromarray(pxls)
    
    if A is None:
        return im
    
    draw = ImageDraw.Draw(im)
    
    # fill in numeric entries if a matrix is provided
    for i in range(0,n):
        for j in range(0,n):
            txt = str(A[i,j])
            w, h = draw.textsize(txt)
            if w > blockSize or h > blockSize:
                continue
            
            x = j * (blockSize + lineSize) + lineSize + blockSize / 2 - w / 2
            y = i * (blockSize + lineSize) + lineSize + blockSize / 2 - h / 2
            draw.text((x,y), txt, fill = 0)
    
    return im


def CreateSDEImage(Q, A = None, blockSize = 24, lineSize = 2, vspace = 36, prefix = None, useColor = True):
    """
    Create an image of SDE stacked vertically
    """
    
    n = len(Q)
    k = np.amax(Q)
    
    if not A is None:
        k = A.shape[0]
    
    sz = n * blockSize + (n+1) * lineSize
    vsz = k * (sz + vspace)
    
    im = Image.new("RGB", (sz, vsz), color=(255,255,255))
    
    draw = ImageDraw.Draw(im)
    
    for i in range(0,k):
        nextimg = []
        if A is None:
            nextimg = CreateSDEMatrixImage(Q, i, None, blockSize, lineSize, useColor)
        else:
            nextimg = CreateSDEMatrixImage(Q, i, A[i,:,:], blockSize, lineSize, useColor)
        
        offset = i * (sz + vspace)
        
        im.paste(nextimg, (0, offset))
        
        if not prefix is None:
            txt = prefix + str(i + 1)
            w, h = draw.textsize(txt)
            draw.text((nextimg.width / 2 - w / 2, offset + nextimg.height + vspace / 2 - h / 2), txt, fill = 0)
    
    return im
        

def CreateSDEPairImage(QA, QX, A = None, X = None, blockSize = 24, lineSize = 2, vspace = 36, hspace = 30, Aprefix = None, Xprefix = None, useColor = True, F = None, T = None):
    """
    Create an image of a pair of SDE stacked vertically
    """
    
    im_A = CreateSDEImage(QA, A, blockSize, lineSize, vspace, Aprefix, useColor)
    im_X = CreateSDEImage(QX, X, blockSize, lineSize, vspace, Xprefix, useColor)
    
    w = im_A.width + hspace + im_X.width
    h = max(im_A.height, im_X.height)
    
    if not F is None and not T is None:
        im_F = CreateSDEMatrixImage(np.zeros(F.shape[0], dtype = int), 1, F, blockSize, lineSize, False)
        im_T = CreateSDEMatrixImage(np.zeros(T.shape[0], dtype = int), 1, T, blockSize, lineSize, False)
        
        w = w + hspace + max(im_F.width, im_T.width)
        h = max(h, im_F.height + im_T.height + vspace)
    
    im = Image.new("RGB", (w, h), color=(255,255,255))
    
    im.paste(im_A, (0,0))
    im.paste(im_X, (im_A.width + hspace, 0))
    
    if not F is None and not T is None:
        im.paste(im_T, (im_A.width + im_X.width + 2 * hspace, 0))
        im.paste(im_F, (im_A.width + im_X.width + 2 * hspace, im_T.height + vspace))
        
        draw = ImageDraw.Draw(im)
        
        txt = 'T'
        w, h = draw.textsize(txt)
        draw.text((im_A.width + im_X.width + 2 * hspace + im_T.width / 2 - w / 2, im_T.height + vspace / 2 - h / 2), txt, fill = 0)
        
        txt = 'F'
        w, h = draw.textsize(txt)
        draw.text((im_A.width + im_X.width + 2 * hspace + im_F.width / 2 - w / 2, im_T.height + im_F.height + vspace * 3 / 2 - h / 2), txt, fill = 0)
    
    return im
    

def ResizeImageUniform(im, width = None, height = None):
    """
    Resize an image by setting a fixed dimension
    """
    
    asp = im.width / im.height
    
    w = im.width
    h = im.height
    
    if not width is None:
        w = width
        h = w / asp
    elif not height is None:
        h = height
        w = h * asp
        
    w = int(w)
    h = int(h)
    
    return im.resize((w, h), resample = 2)


def ExportAsCBF(A, b, name):
    """
    Export an instance to .cbf format compatible in MOSEK
    """
    
    k = A.shape[0]
    n = A.shape[1]
    
    lines = []
    lines.append("VER")
    lines.append("1")
    
    lines.append("PSDVAR")
    lines.append("1")
    lines.append(str(n))
    
    lines.append("CON")
    lines.append(str(k) + " 1")
    lines.append("L= " + str(k))
    
    lines.append("FCOORD")
    elts = len([0 for i in range(0,k) for j in range(0,n) for l in range(j,n) if A[i,j,l] != 0])
    
    lines.append(str(elts))
    for i in range(0,k):
        for j in range(0,n):
            for l in range(j,n):
                if A[i,j,l] != 0:
                    lines.append(str(i) + " 0 " + str(l) + " " + str(j) + " " + str(A[i,j,l]))
    
    lines.append("BCOORD")
    lines.append(str(k))
    for i in range(0,k):
        lines.append(str(i) + " " + str(b[i]))
    
    filename = name + str(".cbf")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for i in lines:
            f.write(i + "\n")
    
    
def ExportAsDATS(A, b, name):
    """
    Export an instance to .dat-s format compatible in SDPA-GMP
    """
    
    k = A.shape[0]
    n = A.shape[1]
    
    lines = []
    lines.append(str(k) + " = mDIM")
    lines.append("1 = nBLOCK")
    lines.append(str(n) + " = bLOCKsTRUCT");
    
    bstring = "{"
    for i in range(0,k):
        bstring += str(b[i])
        if i < k - 1:
            bstring += ", "
    bstring += "}"
    lines.append(bstring)
    
    for i in range(0,k):
        for j in range(0,n):
            for l in range(j,n):
                if A[i,j,l] != 0:
                    lines.append(str(i + 1) + " 1 " + str(j + 1) + " " + str(l + 1) + " " + str(A[i,j,l]))
    
    filename = name + str(".dat-s")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        for i in lines:
            f.write(i + "\n")


def ExportAsMAT(A, b, X, F, T, name):
    """
    Export an instance to .mat format compatible with Matlab
    """
    
    k = A.shape[0]
    l = X.shape[0]
    n = A.shape[1]
    
    Am = np.reshape(A, (k,n*n))
    As = sparse.csr_matrix(Am)
    
    Xl = np.reshape(X, (l,n*n))
    Xs = sparse.csr_matrix(Xl)
    
    filename = name + str(".mat")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    sio.savemat(filename, {'A':As,'b':b,'X':Xs, 'F':F, 'T':T})
    
    
    
    