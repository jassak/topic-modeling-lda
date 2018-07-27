#!/usr/bin/env cython
# coding: utf-8
# cython: embedsignature=True
"""
Cython version of aliassampler.py using functions found in cython/cy_aliassampler.pyx

Created on 25 July 2018

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


class AliasSamplerCy():
    """
    Implements Walker's Alias Method for efficiently sampling from a categorical distribution (biased die),
    see ﻿A. J. Walker. An efficient method for generating discrete random variables with general distributions.

    For the initialization step in particular, it implements Vose's O(n) method, found in
    M. D. Vose. A Linear Algorithm For Generating Random Numbers With a Given Distribution.

    The constructor takes a probability vector and builds the probability and alias tables in O(n) time.
    """

    def __init__(self, prob_vector, dtype=np.float32):
        """
        Calls corresponding cython function

        Args:
            prob_vector: The probability vector must be an array/list of non-negative elements that sum to 1.

            dtype: Data type used during calculations. Chose among np.float16, np.float32 and np.float64.

        """

        self.num_el = len(prob_vector)
        cdef np.ndarray[DTYPE_D_t, ndim=1] prob_table = np.zeros(self.num_el, dtype=DTYPE_D)
        cdef np.ndarray[DTYPE_I_t, ndim=1] alias_table = np.zeros(self.num_el, dtype=DTYPE_I)
        init_tables(self.num_el, self.prob_vector, self.prob_table, self.alias_table)

    def generate(self, n=1):
        """
        Calls corresponding cython function

        Args:
            n:

        Returns:
            A numpy array of size n of random samples from the prob_vector.
        """

        return generate(n, self.num_el, self.prob_table, self.alias_table)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void init_tables(int num_el,
                np.ndarray[DTYPE_D_t, ndim=1] prob_vector,
                np.ndarray[DTYPE_D_t, ndim=1] prob_table,
                np.ndarray[DTYPE_I_t, ndim=1] alias_table):
    """
    TODO comments
    """
    # check consistency of inputs
    assert prob_vector.dtype == DTYPE_D

    # cdef variables
    cdef np.ndarray[DTYPE_D_t, ndim=1] prob_scaled = np.zeros(num_el, dtype=DTYPE_D)
    cdef int i
    cdef int s
    cdef int l
    cdef double p

    # def ques
    # TODO you might want to cythonize the deques
    small = deque()
    large = deque()

    # init variables
    for i in range(num_el):
        alias_table[i] = -1
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

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def generate(int num_samples, int num_el, np.ndarray[DTYPE_D_t, ndim=1] prob_table,
            np.ndarray[DTYPE_I_t, ndim=1] alias_table):
    """
    TODO comments
    """

    # cdef variables
    cdef np.ndarray[DTYPE_I_t, ndim=1] samples = np.zeros(num_samples, dtype=DTYPE_I)
    cdef int i
    cdef int j
    cdef double p

    # generate num_samples using Walker's alias method
    for i in range(num_samples):
        j = random.randrange(num_el)
        # TODO change 1e-5 below to global EPS
        if fabs(prob_table[j] - 1.0) < 1e-5:
            samples[i] = j
        else:
            p = random.random()
            if p <= prob_table[j]:
                samples[i] = j
            else:
                samples[i] = alias_table[j]
    return samples
