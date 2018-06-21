#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12 June 2018

@author: jason
"""

import numpy as np
from collections import defaultdict
from scipy.sparse import dok_matrix, csr_matrix
from aliassampler import AliasSampler


class SparseCounter():
    """Implements a counter object that stores counts in a sparse way, using a dictionary.
    The constructor takes a list of items and counts their occurrences.
    In this implementation counts cannot be negative and exceptions are raised otherwise.
    """

    def __init__(self, seq):
        self.__count = defaultdict()
        for elem in seq:
            if elem in self.__count:
                self.__count[elem] += 1
            else:
                self.__count[elem] = 1

    def get_count(self, key):
        if key in self.__count:
            return self.__count[key]
        else:
            return 0

    def set_count(self, key, count):
        if type(count) != int:
            raise TypeError("Cannot set count to something other than an integer")
        elif count == 0:
            del self.__count[key]
        elif count > 0:
            self.__count[key] = count
        elif count < 0:
            raise ValueError("Cannot set negative count")

    def incr_count(self, key):
        if key in self.__count:
            self.__count[key] += 1
        else:
            self.__count[key] = 1

    def decr_count(self, key):
        if self.get_count(key):
            self.set_count(key, count=self.get_count(key) - 1)
        else:
            raise ValueError("Trying to decrease a zero count")

    def __iter__(self):
        for key in self.__count:
            yield key

    def __len__(self):
        return len(self.__count)


class SparseVector():
    """Implements a sparse one-dimensional vector, encapsulating the cumbersome shape=(1, N) scipy dok_matrix"""

    def __init__(self, vec_size, dtype):
        self.__spmat = dok_matrix((1, vec_size), dtype=dtype)

    def __getitem__(self, item):
        return self.__spmat[0, item]

    def __setitem__(self, key, value):
        self.__spmat[0, key] = value

    def __iter__(self):
        for key in self.__spmat.keys():
            yield key[1]

    def __truediv__(self, other):
        self.__spmat = self.__spmat / other
        return self

    def get_nnz(self):
        return self.__spmat.nnz


class SparseGraph():
    """Sparse graph constructed from a sparse adjacency matrix dense_matr.
    Contains an alias sampler for every node in order to sample nodes according to the probs in dense_matr.
    dot_vec method implements sparse dot multiplication with vector
    in O(average_deg * nnodes) on average."""

    def __init__(self, dense_matr, dtype):
        self.__matr = csr_matrix(dense_matr, dtype=dtype)
        self.nnodes = self.__matr.shape[0]
        # Init neighbourhoods
        self.neighbours = []
        for node in range(self.nnodes):
            self.neighbours.append(self.get_neighbours(node))
        # Init avdeg
        self.avdeg = self.get_average_deg()
        # Init graph alias samplers
        self.aliassamplers = []
        for node in range(self.nnodes):
            neighbs = self.neighbours[node]
            weights = np.zeros(len(neighbs), dtype)
            for (idx, neighb) in enumerate(neighbs):
                weights[idx] = self[node, neighb]
            aliassampler = AliasSampler(weights, dtype)
            self.aliassamplers.append((neighbs, aliassampler))

    def sample_neighbour(self, node):
        neighbs, sampler = self.aliassamplers[node]
        idx = sampler.generate_once()
        return neighbs[idx]

    def get_neighbours(self, node):
        _, neighbs = self.__matr[node].nonzero()
        return neighbs

    def dot_vec(self, vector):
        result = np.zeros(self.nnodes)
        for node in range(self.nnodes):
            result[node] = self.row_dot_vec(node, vector)
        return result

    def row_dot_vec(self, node, vector):
        if len(vector) != self.nnodes:
            raise ValueError("vector dimensions don't match matrix")
        result = 0
        neighbs = self.neighbours[node]
        for neighb in neighbs:
            result += self[node, neighb] * vector[neighb]
        return result

    def get_average_deg(self):
        avdeg = 0
        for node in range(self.nnodes):
            neighbs = self.neighbours[node]
            avdeg += len(neighbs)
        avdeg /= self.nnodes
        return avdeg

    def __getitem__(self, item):
        return self.__matr[item]
