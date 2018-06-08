#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7 June 2018

@author: jason
"""

"""This module contains various general utility classes."""

import logging
import pickle

logger = logging.getLogger(__name__)


class SaveLoad(object):
    """
    TODO Comments
    """

    def save(self, fname, pickle_protocol=3):
        """

        Args:
            fname:
            pickle_protocol:

        Returns:

        """

        with open(fname, 'wb') as file:
            try:
                pickle.dump(self, file, protocol=pickle_protocol)
            except pickle.PicklingError:
                print('unpicklable object')

    @classmethod
    def load(cls, fname):
        """

        Args:
            fname:

        Returns:

        """

        logger.info("loading %s object from %s", cls.__name__, fname)
        with open(fname, 'rb') as file:
            try:
                obj = pickle.load(file)
            except pickle.UnpicklingError:
                print('cannot unpickle')
                return
        logger.info("loaded %s", fname)
        return obj
