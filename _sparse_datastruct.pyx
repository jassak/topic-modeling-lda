#!/usr/bin/env cython
# coding: utf-8
# cython: embedsignature=True

# TODO Cythonize SparseGraph

"""
Created on 17 September 2018

@author: jason
"""

cimport cython
import numpy as np
cimport numpy as np

from collections import defaultdict

from aliassampler import AliasSampler

DTYPE = np.double
ctypedef np.double_t DTYPE_t

DTYPE_TO_EPS = {
    np.float16: 1e-5,
    np.float32: 1e-35,
    np.float64: 1e-100,
}

class SparseCounter():
    """Implements a counter object that stores counts in a sparse way, using a dictionary.
    The constructor takes a list of items and counts their occurrences.
    In this implementation counts cannot be negative and exceptions are raised otherwise.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, list seq):
        self.__count = defaultdict()
        cdef int elem
        for elem in seq:
            if elem in self.__count:
                self.__count[elem] += 1
            else:
                self.__count[elem] = 1

    def get_count(self, int key):
        if key in self.__count:
            return self.__count[key]
        else:
            return 0

    def set_count(self, int key, int count):
        if type(count) != int:
            raise TypeError("Cannot set count to something other than an integer")
        elif count == 0:
            del self.__count[key]
        elif count > 0:
            self.__count[key] = count
        elif count < 0:
            raise ValueError("Cannot set negative count")

    def incr_count(self, int key):
        if key in self.__count:
            self.__count[key] += 1
        else:
            self.__count[key] = 1

    def decr_count(self, int key):
        if self.get_count(key):
            self.set_count(key, count=self.get_count(key) - 1)
        else:
            raise ValueError("Trying to decrease a zero count")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __iter__(self):
        cdef int key
        for key in self.__count:
            yield key

    def __len__(self):
        return len(self.__count)

class SparseVector():
    """
    Implements a sparse one-dimensional vector equipped with normalize method
    (replacing very slow version using scipy.dok_matrix)
    """


    def __init__(self, int vec_size, dtype=DTYPE):
        self.__dict = defaultdict()
        self.vec_size = vec_size
        cdef int nnz
        self.nnz = nnz
        self.dtype = dtype

    def __getitem__(self, int key):
        if 0 <= key < self.vec_size:
            try:
                return self.__dict[key]
            except:
                return 0
        else:
            raise IndexError('index out of bounds')

    def __setitem__(self, int key, value):
        if 0 <= key < self.vec_size:
            if not value == 0:
                self.__dict[key] = value
                self.nnz += 1
            else:
                del self.__dict[key]
                self.nnz -= 1
        else:
            raise IndexError('index out of bounds')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __iter__(self):
        cdef int key
        for key in self.__dict:
            yield key

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def normalize(self):
        cdef double vecsum = 0.0
        cdef int key
        for key in self.__dict:
            vecsum += self.__dict[key]
        if abs(vecsum) > DTYPE_TO_EPS[self.dtype]:
            for key in self.__dict:
                self.__dict[key] /= vecsum
        else:
            raise ValueError('cannot normalize zero vector')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def make_weight_vec(self):
        cdef np.ndarray[DTYPE_t, ndim=1] weight_vec = np.zeros(self.nnz, dtype=self.dtype)
        cdef list index_map = [0] * self.nnz
        cdef int idx
        cdef int key
        for idx, key in enumerate(self.__dict):
            weight_vec[idx] = self.__dict[key]
            index_map[idx] = key
        return index_map, weight_vec

    def get_nnz(self):
        return self.nnz
