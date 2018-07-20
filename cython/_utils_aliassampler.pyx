#!/usr/bin/env cython
# coding: utf-8
# cython: embedsignature=True
"""
Created on 20 July 2018

@author: jason
"""

import random
from collections import deque

cdef void init_tables(int num_el, double * prob_vector):
    """
    TODO comments
    """
    cdef double prob_table[num_el]
    cdef int alias_table[num_el]
    for int i in range(num_el):
        alias_table[i] = -1


    small = deque
    large = deque

    # TODO CONTINUE HERE
