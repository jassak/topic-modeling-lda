#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7 June 2018

@author: jason
"""

"""This module contains the base topic model from which all topic models inherit"""

from utils import SaveLoad
from abc import ABC, abstractmethod


class ABCTopicModel(ABC, SaveLoad):
    """
    TODO Comments
    """

    def print_topic_terms(self, topic_id, topn=10, with_prob=False):
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
