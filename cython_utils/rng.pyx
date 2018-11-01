#!/usr/bin/env cython
# coding: utf-8

"""
Created on 1 November 2018

@author: jason
"""

cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport malloc, free, rand, RAND_MAX, srand

@cython.cdivision(True)
cdef double randUniform() nogil:
    return <double> rand() / RAND_MAX

cdef int randInt(int low, int high) nogil:
    return <int> floor((high - low) * randUniform() + low)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int rand_choice(int n, double * prob) nogil:
    cdef int i
    cdef double r
    cdef double cuml
    r = <double> rand() / RAND_MAX
    cuml = 0.0
    for i in range(n):
        cuml = cuml + prob[i]
        if (r <= cuml):
            return i
