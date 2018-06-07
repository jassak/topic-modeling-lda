#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7 June 2018

@author: jason
"""

"""This module contains the base topic model from which all topic models inherit"""

from utils import SaveLoad

class BaseTopicModel(SaveLoad):
    """
    TODO Comments
    """
    def print_topic_terms(self, topic_id, topn=10, with_prob=False):
        """

        Args:
            topic_id:
            topn:

        Returns:

        """

        print('Printing terms in topic number', topic_id)
        if with_prob:
            raise NotImplementedError
        else:
            print('\t' + ' + '.join([term[0] for term in self.get_topic_terms(topic_id, topn)]))

    def print_document_topics(self, doc_id, minimum_probability=None, with_prob=False):
        """

        Args:
            doc_id:
            minimum_probability:

        Returns:

        """
        print('Printing topics in document number', doc_id)
        if with_prob:
            raise NotImplementedError
        else:
            raise NotImplementedError
            # print('\n'.join(self.print_topic_terms(topic_id)
            #                 for (topic_id, _) in self.get_document_topics(doc_id, readable=False)))