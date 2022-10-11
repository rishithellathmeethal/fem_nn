import numpy as np
import pylab as py
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class fem():
    def __init__(self):
        pass

    def make_a_matrix(self, size, del_h):
        A = np.zeros((size, size))
        for i in range(size):
            if i==0 or i==size-1:
                A[i][i] = 1   
                if i==0:  A[i][i+1] = -1 
                if i==size-1: A[i][i-1] = -1 
            else:
                A[i][i], A[i][i+1], A[i][i-1] = 2, -1, -1
        return (1/del_h)* A

    def make_f_matrix(self, size, del_h, f):
        return f*del_h* np.ones((size, 1))

    def apply_BC(self, A, F, left, right):
        F[0], F[-1] = left, right
        A[0][0], A[0][1], A[-1][-1], A[-1][-2] = 1, 0, 1, 0
        return A, F