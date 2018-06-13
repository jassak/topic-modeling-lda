#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO change ints to floats in various probabilities

Created on 21 May 2018

@author: jason
"""

import numpy as np
import random
from collections import deque

DTYPE_TO_EPS = {
    np.float16: 1e-5,
    np.float32: 1e-35,
    np.float64: 1e-100,
}


class AliasSampler():
    """
    Implements Walker's Alias Method for efficiently sampling from a categorical distribution (biased die),
    see ﻿A. J. Walker. An efficient method for generating discrete random variables with general distributions.

    For the initialization step in particular, it implements Vose's O(n) method, found in
    M. D. Vose. A Linear Algorithm For Generating Random Numbers With a Given Distribution.

    The constructor takes a probability vector and builds the probability and alias tables in O(n) time.
    """

    def __init__(self, prob_vector, dtype=np.float32):
        """

        Args:
            prob_vector: The probability vector must be an array/list of non-negative elements that sum to 1.

            dtype: Data type used during calculations. Chose among np.float16, np.float32 and np.float64.

        """

        if dtype not in DTYPE_TO_EPS:
            raise ValueError(
                    "Incorrect 'dtype', please choose one of {}".format(
                            ", ".join("numpy.{}".format(tp.__name__) for tp in sorted(DTYPE_TO_EPS))))
        self.dtype = dtype
        eps = DTYPE_TO_EPS[self.dtype]

        self.prob_vector = np.asarray(prob_vector, self.dtype)
        assert (np.amin(prob_vector) >= 0.), "probabilities are not non-negative"
        # assert (abs(np.sum(prob_vector) - 1.) <= eps), "probabilities must sum to 1"
        assert (prob_vector.ndim == 1), "a probability vector must have dimension 1"

        num_el = len(prob_vector)
        self.num_el = num_el

        prob_table = np.empty(num_el, dtype=self.dtype)
        self.prob_table = prob_table

        alias_table = [None] * num_el
        self.alias_table = alias_table

        small = deque()
        large = deque()
        prob_scaled = prob_vector * num_el

        for idx, p in enumerate(prob_scaled):
            if p < 1:
                small.append(idx)
            else:
                large.append(idx)
        while small and large:
            s = small.pop()
            l = large.pop()
            prob_table[s] = prob_scaled[s]
            alias_table[s] = l
            prob_scaled[l] = (prob_scaled[s] + prob_scaled[l]) - 1
            if prob_scaled[l] < 1:
                small.append(l)
            else:
                large.append(l)
        while large:
            l = large.pop()
            prob_table[l] = 1
        while small:
            s = small.pop()
            prob_table[s] = 1

    def generate(self, n=1):
        """

        Args:
            n:

        Returns:
            A numpy array of size n of random samples from the prob_vector.
        """

        samples = [0] * n
        for i in range(n):
            samples[i] = self.generate_once()

        return samples

    def generate_once(self):
        """

        Returns:
            A random sample from the prob_vector in O(1) time, once prob_table and alias_table have been built.
        """

        i = random.randrange(self.num_el)
        if self.prob_table[i] == 1:
            return i
        else:
            coin = random.random()
            if coin <= self.prob_table[i]:
                return i
            else:
                return self.alias_table[i]
