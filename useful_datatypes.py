#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12 June 2018

@author: jason
"""

from collections import defaultdict


class SparseCounter():
    """Implements a counter object that stores counts in a sparse way, using a dictionary.
    The constructor takes a list of items and counts their occurrences.
    In this implementation counts cannot be negative and exceptions are raise otherwise.
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
        self.__count[key] += 1

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
