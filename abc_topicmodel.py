#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7 June 2018

@author: jason
"""
import numbers
import numpy as np
import six

from utilityclasses import SaveLoad
from abc import ABC, abstractmethod

"""This module contains the abstract base class topic model from which all topic models inherit.
It implements various printing methods, common to all topic models."""


class ABCTopicModel(ABC, SaveLoad):
    """
    TODO Comments
    """

    def print_topic_terms(self, topic_id, topn=10, with_prob=True):
        """

        Args:
            topic_id:
            topn:
            with_prob:

        Returns:

        """

        print('Printing terms in topic', topic_id)
        if with_prob:
            print('\t' + ' + '.join(['@'.join(list(map(str, term))) for term in self.get_topic_terms(topic_id, topn)]))
        else:
            print('\t' + ' + '.join([term[0] for term in self.get_topic_terms(topic_id, topn)]))

    def print_term_topics(self, term_id, topn=10, with_prob=True):

        print('Printing topics for term', term_id)
        if with_prob:
            print('\t' + ' + '.join([''.join(['topic', str(topic[0]), '@', str(topic[1])])
                                     for topic in self.get_term_topics(term_id, topn=topn, minimum_prob=0)]))
        else:
            print('\t' + ' + '.join([''.join(['topic', str(topic[0])])
                                     for topic in self.get_term_topics(term_id, topn=topn, minimum_prob=0)]))

    def print_document_topics(self, doc_id, minimum_prob=None, with_prob=True):
        """

        Args:
            doc_id:
            minimum_prob:
            with_prob:

        Returns:

        """
        print('Printing topics in document', doc_id)
        if with_prob:
            for (topic_id, prob) in self.get_document_topics(doc_id, readable=False):
                print('\t', ''.join(['topic', str(topic_id)]), '{',
                      ' + '.join([term[0] for term in self.get_topic_terms(topic_id, 10)]),
                      '} @', '{:.6f}'.format(prob))
        else:
            for (topic_id, _) in self.get_document_topics(doc_id, readable=False):
                print('\t', ''.join(['topic', str(topic_id)]), '{',
                      ' + '.join([term[0] for term in self.get_topic_terms(topic_id, 10)]),
                      '}')

    def print_topic_documents(self, topic_id, topn=10, minimum_prob=0, with_prob=True):

        print('Printing documents for topic', topic_id)
        if with_prob:
            print('\t' + ' + '.join([''.join(['doc', str(doc[0]), '@', str(doc[1])])
                                     for doc in self.get_topic_documents(topic_id, topn=topn, minimum_prob=0)]))
        else:
            print('\t' + ' + '.join([''.join(['doc', str(doc[0])])
                                     for doc in self.get_topic_documents(topic_id, topn=topn, minimum_prob=0)]))

    def init_dir_prior(self, num_topics, num_terms, dtype, prior, name):
        """
        Initializes the Dirichlet priors. Copied from gensim.

        Args:
            prior:
            name:

        Returns:

        """

        if prior is None:
            prior = 'symmetric'

        if name == 'alpha':
            prior_shape = num_topics
        elif name == 'beta':
            prior_shape = num_terms
        else:
            raise ValueError("'name' must be 'alpha' or 'beta'")

        is_auto = False

        # TODO Something is wrong here, I think it assigns beta = 1/num_topics for prior=symmetric
        if isinstance(prior, six.string_types):
            if prior == 'symmetric':
                # logger.info("using symmetric %s at %s", name, 1.0 / num_topics)
                init_prior = np.asarray([1.0 / num_topics for _ in range(prior_shape)], dtype=dtype)
            elif prior == 'asymmetric':
                init_prior = \
                    np.asarray([1.0 / (i + np.sqrt(prior_shape)) for i in range(prior_shape)], dtype=dtype)
                init_prior /= init_prior.sum()
                # logger.info("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                # This is obviously wrong since it's the same as symmetric. Maybe in future correct it.
                init_prior = np.asarray([1.0 / num_topics for _ in range(prior_shape)], dtype=dtype)
                # if name == 'alpha':
                    # logger.info("using autotuned %s, starting with %s", name, list(init_prior))
            else:
                raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior, dtype=dtype)
        elif isinstance(prior, np.ndarray):
            init_prior = prior.astype(dtype, copy=False)
        elif isinstance(prior, np.number) or isinstance(prior, numbers.Real):
            init_prior = np.asarray([prior] * prior_shape, dtype=dtype)
        else:
            raise ValueError("%s must be either a np array of scalars, list of scalars, or scalar" % name)

        return init_prior, is_auto

    @abstractmethod
    def get_topic_terms(self, topic_id, topn, readable):
        raise NotImplementedError

    @abstractmethod
    def get_term_topics(self, term_id, topn, minimum_prob):
        raise NotImplementedError

    @abstractmethod
    def get_document_topics(self, doc_id, minimum_prob, readable):
        raise NotImplementedError

    @abstractmethod
    def get_topic_documents(self, topic_id, topn, minimum_prob):
        raise NotImplementedError