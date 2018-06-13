#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 13 June 2018

@author: jason
"""

import unittest
from aliassampler import AliasSampler


class TestAliasSampler(unittest.TestCase):

    def test_constructor(self):
        bad_prob_vector = [0.3, 0.5, 0.2, 0.1]
        self.assertRaises(ValueError, AliasSampler, bad_prob_vector)

    def test_generate(self):
        pass

    def test_generate_once(self):
        pass
