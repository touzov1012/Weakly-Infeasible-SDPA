# -*- coding: utf-8 -*-
"""
Example showing how to run the bilinear algorithm

@author: alex
"""

import numpy as np

from wsdp_utility import *
from wsdp_export import *
from wsdp_alg_base import *
from wsdp_alg_bilinear import *

np.random.seed(19)

# Create SDE structure from weak SDP by defining sizes of each consecutive
# index block P_i (including the left over region).
P = CreateConsecPartition([2,1,3,2,1,3])

# Create a random SDE structure for the certificate of the above weak SDP
Q = CreateCertificateStructure(5, P)

# Check if the certificate and weak SDP SDE structures are compatible.
isValid = ValidateCertStructure(P, Q)
print("Valid P,Q: " + str(isValid))

# Execute the bilinear algorithm to find a weak SDP and certificate with the
# structures specified above.
[A, b, X] = CreateWeakSysCertSDE(P, Q, [-2,2], [1,1])

# Check if the generated system and certificate are valid. This should always
# be true, use this function for debugging.
isValid = ValidateWeakCert(A, b, X, 1, True)
print("Valid: " + str(isValid))

# Create an image of the system and certificate and rescale it uniformly.
img = CreateSDEPairImage(P, Q, A, X, Aprefix='A', Xprefix='X')
img = ResizeImageUniform(img, width = 1024)

# Extend the SDE of the above weak SDP with additional entries.
[A, b] = ExtendWeakSDE(A, b, X, 4)

# Check that the extended sequence is valid.
isValid = ValidateWeakCert(A, b, X, 1, True)
print("Valid: " + str(isValid))

# Display the maximum elements of the system A and certificate X
print("A max: " + str(np.amax(A)))
print("X max: " + str(np.amax(X)))

# Check the condition number of the operator A
print("A condition: " + str(Condition(A)))

# Create an image of the system and certificate and rescale it uniformly.
img_extended = CreateSDEPairImage(P, Q, A, X, Aprefix='A', Xprefix='X')
img_extended = ResizeImageUniform(img_extended, width = 1024)

# Rotate elements of A and X arbitrarily
[T,Ti] = RotateBases(A, X, 100)

# Check that the rotated sequence is correct.
isValid = ValidateWeakCert(A, b, X, 1, True)
print("Valid: " + str(isValid))

# Rotate the entire sequence A using row operations
[A, b, F] = RotateSequence(A, b)

# Check that the rotated sequence is correct.
isValid = ValidateWeakCert(A, b, X, 1, True)
print("Valid: " + str(isValid))

# Check the condition number of the operator A
print("A condition: " + str(Condition(A)))

# Create an image of the system and certificate and rescale it uniformly.
img_messy = CreateSDEPairImage(P, Q, A, X, Aprefix='A', Xprefix='X', useColor=False, F=F, T=T)
img_messy = ResizeImageUniform(img_messy, width = 1024)