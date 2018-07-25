#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cython version of aliassampler.py using functions found in cython/cy_aliassampler.pyx

Created on 25 July 2018

@author: jason
"""

import cy_aliassampler
import numpy as np

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
        self.prob_table, self.alias_table = cy_aliassampler.init_tables(self.num_el, prob_vector)

    def generate(self, n=1):
        """
        Calls corresponding cython function

        Args:
            n:

        Returns:
            A numpy array of size n of random samples from the prob_vector.
        """

        return cy_aliassampler.generate(n, self.num_el, self.prob_table, self.alias_table)
