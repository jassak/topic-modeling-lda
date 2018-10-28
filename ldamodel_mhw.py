#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 28 October 2018

@author: jason
"""

# TODO use numpy arrays for theta and phi, as in deprecated version, because get_term_topics
# and get_document_topics are broken (no transpose method for python lists).

from abc_topicmodel import ABCTopicModel
from _ldamhw_train import train
from gensim import matutils

from profiling_utils import profileit


class LDAModelMHW(ABCTopicModel):
    """
    The constructor estimates Latent Dirichlet Allocation model parameters based
    on a training corpus, according to the Metropolis-Hasting-Walker variant of
    Gibbs sampling method described in
    **Li, Ahmed, Ravi, Smola: ﻿Reducing the sampling complexity of topic models. ﻿KDD 2014.**

    Example:
        lda = LDAModelMHW(corpus, num_topics=10)
    """

    def __init__(self, corpus=None, num_topics=100, num_passes=10, minimum_prob=0.01):
        """

        Args:
            corpus: If given, start training from the iterable `corpus` straight away. If not given,
                the model is left untrained (presumably because you want to call `train()` manually).

            num_topics: The number of requested latent topics to be extracted from
                the training corpus.

            num_passes: The number of passes of the MCMC procedure. One pass is one step per term
                in each document of the whole corpus.

        """

        # store user-supplied parameters
        if corpus is not None:
            self.id2word = corpus.dictionary
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.id2word = None
            self.num_terms = 0

        self.num_topics = int(num_topics)
        self.minimum_probability = minimum_prob

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.theta, self.phi = train(self.num_topics, num_passes, corpus)

    def get_topic_terms(self, topic_id, topn=10, readable=True):
        # TODO move this and similar methods to parent class
        """

        Args:
            topic_id:
            topn:
            readable: If False returns term_id, if True returns the actual word.

        Returns:
             A list of tuples (term, prob) of the topn terms in topic_id, formated according to format.

        """

        topic_term_probs = self.phi[topic_id]
        bestn = matutils.argsort(topic_term_probs, topn, reverse=True)
        if readable:
            return [(self.id2word[idx], topic_term_probs[idx]) for idx in bestn]
        else:
            return [(idx, topic_term_probs[idx]) for idx in bestn]

    def get_term_topics(self, term_id, topn=10, minimum_prob=0):
        """

        Args:
            term_id:
            topn:
            minimum_prob:

        Returns:
            A list of tuples (topic, prob) of topics containing term_id with prob greater than minimum_prob.

        """

        term_topic_probs = self.phi.transpose()[term_id]
        sorted_probs = matutils.argsort(term_topic_probs, topn=topn, reverse=True)
        return [(topic_id, term_topic_probs[topic_id]) for topic_id in sorted_probs
                if term_topic_probs[topic_id] > minimum_prob]

    def get_document_topics(self, doc_id, minimum_prob=0, readable=True):
        """

        Args:
            doc_id:
            minimum_prob: Ignore topics below this probability.
            readable: If False returns topic_id's. Else returns a string of the top 10 words in topic.

        Returns:
            A list of tuples (topic, probability) for document[doc_id]. topic is either topic_id or a string.

        """

        if minimum_prob is None:
            minimum_prob = self.minimum_probability
        minimum_prob = max(minimum_prob, 1e-8)  # never allow zero values in sparse output. (Why??)

        doc_topic_probs = self.theta[doc_id]
        sorted_idx = matutils.argsort(doc_topic_probs, reverse=True)
        if not readable:
            return [(idx, doc_topic_probs[idx]) for idx in sorted_idx if doc_topic_probs[idx] > minimum_prob]
        else:
            doc_topics = []
            for idx in sorted_idx:
                if doc_topic_probs[idx] > minimum_prob:
                    topic_terms = self.get_topic_terms(idx, topn=10, readable=True)
                    terms_string = " ".join([w[0] for w in topic_terms])
                    doc_topics.append((terms_string, doc_topic_probs[idx]))
            return doc_topics

    def get_topic_documents(self, topic_id, topn=10, minimum_prob=0):
        """

        Args:
            topic_id:
            minimum_prob:

        Returns:
            A list of tuples (doc_id, probability) of documents containing topic_id with prob greater than minimum_prob.

        """

        topic_docs_probs = self.theta.transpose()[topic_id]
        sorted_probs = matutils.argsort(topic_docs_probs, topn=topn, reverse=True)
        return [(doc_id, topic_docs_probs[doc_id]) for doc_id in sorted_probs
                if topic_docs_probs[doc_id] > minimum_prob]
