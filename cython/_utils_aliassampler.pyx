#!/usr/bin/env cython
# coding: utf-8
# cython: embedsignature=True
"""
Created on 20 July 2018

@author: jason
"""
import numpy as np
import random
from collections import deque
import cython

from libc.math cimport fabs
cimport numpy as np

DTYPE_D = np.double
ctypedef np.double_t DTYPE_D_t
DTYPE_I = np.int
ctypedef np.int_t DTYPE_I_t

def init_tables(int num_el, np.array[DTYPE_D_t, ndim=1] prob_vector):
    """
    TODO comments
    """
    # check consistency of inputs
    assert prob_vector.dtype == DTYPE

    # cdef variables
    cdef np.ndarray[DTYPE_D_t, ndim=1] prob_table = np.zeros(num_el, dtype=DTYPE_D)
    cdef np.ndarray[DTYPE_D_t, ndim=1] prob_scaled = np.zeros(num_el, dtype=DTYPE_D)
    cdef np.ndarray[DTYPE_I_t, ndim=1] alias_table = np.zeros(num_el, dtype=DTYPE_I)
    cdef int i
    cdef int s
    cdef int l
    cdef double p

    # def ques
    # TODO someday you might want to cythonize the deques
    small = deque()
    large = deque()

    # init variables
    for int i in range(num_el):
        alias_table[i] = None
        prob_scaled[i] = prob_vector[i] * <double>num_el

    # divide prob in small and large
    for i in range(num_el):
        p = prob_scaled[i]
        if p < 1.0:
            small.append(i)
        else:
            large.append(i)

    # take from large and give to small (main idea)
    while small and large:
        s = small.pop()
        l = large.pop()
        prob_table[s] = prob_scaled[s]
        alias_table[s] = l
        prob_scaled[l] = (prob_scaled[s] + prob_scaled[l]) - 1.0
        if prob_scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    # pad 1s to small and large
    while large:
        l = large.pop()
        prob_table[l] = 1.0
    while small:
        s = small.pop()
        prob_table[s] = 1.0

    # TODO CONTINUE HERE
